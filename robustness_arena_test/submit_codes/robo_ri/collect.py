from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .arena import activate
from .calibration import save_pair
from .perturbations import Perturbation
from .ri_mem.evidence import EvidenceIndex, MemoryKey, patch_keys
from .ri_mem.query import Plan, QwenService
from .ri_mem.tracking import RegionTracker
from .runner import atomic_json


class EvidenceStream:
    def __init__(self, service, plan, capacity):
        self.service, self.plan = service, plan
        self.index, self.tracker = EvidenceIndex(capacity), RegionTracker()
        self.current = []

    def observe(self, image, step):
        if step % 80 == 0:
            self.current = self.service.detect(image, self.plan.entities, step)
            self.tracker.seed(image, self.current)
        else:
            self.current = self.tracker.update(image, step)
        self.index.add(step, self.current)

    def features(self, keys, stage, step):
        return self.index.features(keys, self.current, self.plan.entities[stage["t"]], stage["e"], step)


def simplemem_batch(policy, stream, observation, prompt, stage, step, actions):
    import torch
    from robomemarena_sim.policy_adapter import encode_state
    from simplememvla.data.collator import SimpleMemVLADataCollator
    from simplememvla.data.messages import build_simplememvla_messages
    from simplememvla.benchmarks.robomemarena import CAMERA_LABELS
    videos, metas, images = policy._visual_inputs()
    action = torch.as_tensor(actions, dtype=torch.float32)
    normalized = policy.unnormalize.normalize(action)
    state = torch.as_tensor(encode_state(observation["_raw_obs"]))
    if policy.use_proprio:
        state = policy.normalize_state.normalize(state)
    feature = {"messages": build_simplememvla_messages(policy.camera_tags, prompt, policy.robot_tag,
        policy.control_frequency_hz, policy.action_horizon, CAMERA_LABELS, subtask=stage["d"],
        history_camera_tags=policy.history_camera_tags), "videos": videos, "video_metadata": metas,
        "images": images, "actions": normalized, "action_mask": torch.ones_like(normalized, dtype=torch.bool),
        "state": state}
    inputs = SimpleMemVLADataCollator(policy.processor)([feature])
    token = policy.vla.backbone.config.video_token_id
    positions = (inputs["input_ids"][0] == token).nonzero().flatten().numpy()
    features = np.zeros((*inputs["input_ids"].shape, 4), np.float32)
    historical = np.zeros(inputs["input_ids"].shape, bool)
    merge = policy.vla.backbone.config.vision_config.spatial_merge_size
    cursor = 0
    for camera, (temporal, h, w) in zip(policy.history_image_keys, inputs["video_grid_thw"].tolist(), strict=True):
        h, w = h // merge, w // merge
        offset = policy._frames_seen[camera] - len(policy._buffers[camera])
        frames = [offset + i for i in policy._history_positions(camera)]
        if len(frames) != temporal * 2:
            raise ValueError("Unexpected native temporal merge")
        for t in range(temporal):
            boxes = [(x / w, y / h, (x + 1) / w, (y + 1) / h) for y in range(h) for x in range(w)]
            contributions = [stream.features([MemoryKey(f, b, f < step) for b in boxes], stage, step)
                             for f in frames[t * 2:t * 2 + 2]]
            index = positions[cursor:cursor + h * w]
            features[0, index] = np.mean(contributions, axis=0)
            historical[0, index] = any(f < step for f in frames[t * 2:t * 2 + 2])
            cursor += h * w
    if cursor != len(positions):
        raise ValueError("Video key layout mismatch")
    return {"inputs": {k: v.detach().cpu().numpy() for k, v in inputs.items() if k != "actions"},
            "actions": normalized.numpy()[None], "features": features, "historical": historical}


def fs_batch(policy, stream, observation, prompt, stage, step, actions):
    inputs = {k: observation[k] for k in ("observation/image", "observation/wrist_image", "observation/state")}
    inputs.update(prompt=prompt, actions=actions)
    inputs = policy._input_transform(policy._prepare_history(inputs))
    frames = policy.mem_buffer.get_frame_sampling_indices(policy.step_idx, policy.config.budget, 16)
    values = stream.features(patch_keys(frames, 4, step), stage, step)
    features = np.zeros((1, policy.config.budget, 4), np.float32)
    features[0, inputs["static_mask"].astype(bool)] = values
    normalized = np.asarray(inputs.pop("actions"))[None]
    def batch(tree):
        return {k: batch(v) if isinstance(v, dict) else np.asarray(v)[None]
                for k, v in tree.items() if v is not None}
    return {"inputs": batch(inputs), "actions": normalized, "features": features}


def collect(manifest, output, paths, method):
    """Render paired expert states; only distractors are used for head calibration."""
    import h5py
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    data = json.loads(manifest.read_text())
    if data.get("split") != "train":
        raise ValueError("Only official training replays are accepted")
    ec, generic = activate(paths["arena_repo"], paths["cache"])
    service = QwenService(str(paths["qwen_checkpoint"]), paths.get("qwen_device", "cuda:0"))
    if method == "fs_mod":
        from .paths import activate_fs
        activate_fs(paths)
        from mme_vla_suite.policies.policy_config import create_trained_policy
        from mme_vla_suite.training.config import get_config
        directory = paths["cache"] / "fs_checkpoint"
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "56000"
        if not checkpoint.exists():
            checkpoint.symlink_to(paths["fs_checkpoint"], target_is_directory=True)
        (directory / "history_config.txt").write_text("perceptual-framesamp-modul.yaml")
        base = create_trained_policy(get_config("mme_vla_ftv2_all26_framesamp_modul"), checkpoint, seed=7)
        policies = [copy.copy(base), copy.copy(base)]
        for policy in policies:
            policy.reset()
    else:
        from .baselines import SimpleMem
        base = SimpleMem(paths).adapter.policy
        policies = [copy.copy(base), copy.copy(base)]
        for policy in policies:
            policy._buffers = {k: [] for k in base.image_keys}
            policy.reset()
    records = []
    for trajectory_index, row in enumerate(data["trajectories"]):
        condition = row["condition"]
        if condition not in ("tv_on", "object_passby", "unrelated_object", "target_like_object", "camera_shake", "light_change"):
            raise ValueError("Calibration uses paired clean/distractor renders")
        source = (manifest.parent / row["file"]).resolve()
        with h5py.File(source, "r") as handle:
            group = handle[row["group"]]
            states, expert_actions = np.asarray(group["states"]), np.asarray(group["actions"])
        if expert_actions.ndim != 2 or expert_actions.shape[1] != 7:
            raise ValueError("Expected Arena expert actions [steps, 7]")
        task, seed = int(row["task"]), int(row["seed"])
        plan = Plan.parse(row["plan"])
        stage_steps = row["stage_start_steps"]
        if len(stage_steps) != len(plan.subgoals) or stage_steps[0] != 0 or sorted(set(stage_steps)) != stage_steps:
            raise ValueError("Provide ordered expert stage boundaries for every subgoal")
        for policy in policies:
            policy.reset()
        streams = [EvidenceStream(service, plan, 256 if method == "fs_mod" else 16384) for _ in range(2)]
        np.random.seed(seed)
        env = ec._get_env_class()(bddl_file_name=str(ec._resolve_bddl_path(task)), camera_heights=256,
            camera_widths=256, ignore_done=True, reward_shaping=True, control_freq=20, initialization_noise=None)
        env.seed(seed)
        event = Perturbation(condition, task, seed)
        event.install(env)
        selected = set(row["sample_steps"])
        try:
            env.reset()
            def visual_state():
                return {"geom_rgba": event.m.geom_rgba.copy(), "cam_pos": event.m.cam_pos.copy(),
                        "light_ambient": event.m.light_ambient.copy(), "light_diffuse": event.m.light_diffuse.copy(),
                        "mocap_pos": event.d.mocap_pos.copy(), "mocap_quat": event.d.mocap_quat.copy()}
            def restore_visual(state):
                for key, value in state.items():
                    getattr(event.d if key.startswith("mocap") else event.m, key)[:] = value
            clean_visual = visual_state()
            clean_visual["mocap_pos"][:] = [-3, 0, 2]
            clean_visual["geom_rgba"][[env.env.sim.model.geom_name2id(f"ri_stripe_{i}") for i in range(8)], 3] = 0
            _, task_key = ec._resolve_task_id(task)
            prompt = ec.get_prompt(task_key, ec._resolve_bddl_path(task).stem)
            for step in range(max(selected) + 1):
                if step >= len(states):
                    raise ValueError("Calibration step exceeds demonstration length")
                perturbed_visual = visual_state()
                restore_visual(clean_visual)
                clean_obs = env.regenerate_obs_from_state(states[step])
                clean_obs = {k: np.array(v, copy=True) for k, v in clean_obs.items()}
                restore_visual(perturbed_visual)
                event.s = step + 10
                event._tick_distractor()
                perturbed_obs = env.regenerate_obs_from_state(states[step])
                index = max(i for i, start in enumerate(stage_steps) if start <= step)
                stage = plan.subgoals[index]
                batches = []
                for policy, stream, raw in zip(policies, streams, (clean_obs, perturbed_obs)):
                    obs, front, wrist = generic.build_eval26_policy_input(raw_obs=raw, prompt=prompt, resize_size=256)
                    stream.observe(front, step)
                    if method == "fs_mod":
                        policy.add_buffer({"images": np.asarray(front)[None, None],
                                           "state": np.asarray(obs["observation/state"])[None], "exec_start_idx": 0})
                        horizon = 20
                    else:
                        from robomemarena_sim.policy_adapter import SimpleMemVLARoboMemArenaAdapter
                        policy.observe(SimpleMemVLARoboMemArenaAdapter._processed_frames(obs))
                        horizon = policy.action_horizon
                    if step in selected:
                        if step + horizon > len(expert_actions):
                            raise ValueError("Choose sample steps with a full expert action horizon")
                        action = expert_actions[step:step + horizon]
                        prepare = fs_batch if method == "fs_mod" else simplemem_batch
                        sample = prepare(policy, stream, obs, prompt, stage, step, action)
                        sample["query"] = np.array([[float(c in stage["e"]) for c in "SIPM"] + [1., 1.]], np.float32)
                        batches.append(sample)
                if batches:
                    pair_id = f"t{task:02d}_demo{trajectory_index}_step{step}"
                    records.extend(save_pair(output, pair_id, *batches, states[step], stage["d"], condition))
        finally:
            event.close()
            env.close()
    atomic_json(output / "manifest.json", {"split": "train", "method": method,
                                           "source_manifest": str(manifest), "samples": records})
    return output / "manifest.json"
