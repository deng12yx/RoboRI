from __future__ import annotations

import json
import sys

import numpy as np


class Pi05:
    def __init__(self, host="127.0.0.1", port=8800):
        self.client = LocalClient(host, port)

    def reset(self):
        pass

    def infer_actions(self, obs, prompt, resize_size):
        keys = ("observation/image", "observation/wrist_image", "base_0_rgb", "left_wrist_0_rgb",
                "right_wrist_0_rgb", "observation/state")
        return np.asarray(request(self.client, {**{k: obs[k] for k in keys if k in obs}, "prompt": prompt})["actions"])

    def close(self):
        self.client._ws.close()


def request(client, payload):
    from openpi_client.msgpack_numpy import unpackb
    client._ws.send(client._packer.pack(payload))
    reply = client._ws.recv(timeout=180)
    if isinstance(reply, str):
        raise RuntimeError(reply)
    result = unpackb(reply)
    if result.get("error") == "invalid_query_plan":
        from .ri_mem.query import InvalidPlan
        raise InvalidPlan(result["message"])
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result


class LocalClient:
    def __init__(self, host, port):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        self._packer = msgpack_numpy.Packer()
        self._ws = connect(f"ws://{host}:{int(port)}", proxy=None, compression=None, max_size=None,
                           ping_interval=None, ping_timeout=None, open_timeout=30)
        self.metadata = msgpack_numpy.unpackb(self._ws.recv(timeout=30))

    def get_server_metadata(self):
        return self.metadata


class FrameSample:
    def __init__(self, host="127.0.0.1", port=8801, expected_method="fs_mod"):
        self.client = LocalClient(host, port)
        if self.client.get_server_metadata().get("method") != expected_method:
            self.client._ws.close()
            raise RuntimeError(f"Expected {expected_method} server; metadata: {self.client.get_server_metadata()}")
        self.frames = []

    def reset(self):
        self.frames.clear()
        self.first = True
        self.client._ws.send(self.client._packer.pack({"reset": True}))
        from openpi_client.msgpack_numpy import unpackb
        reply = self.client._ws.recv(timeout=180)
        if isinstance(reply, str) or not unpackb(reply).get("reset_finished"):
            raise RuntimeError(f"FS+Mod reset failed: {reply!r}")

    def observe(self, obs, prompt, resize_size):
        self.frames.append((obs["observation/image"], obs["observation/wrist_image"], obs["observation/state"]))

    def infer_actions(self, obs, prompt, resize_size):
        if not self.frames:
            raise RuntimeError("Empty memory input")
        front, wrist, state = zip(*self.frames)
        result = request(self.client, {"front_rgb_list": list(front), "wrist_rgb_list": list(wrist),
            "joint_state_list": [np.asarray(s)[:7] for s in state],
            "gripper_state_list": [np.asarray(s)[7:8] for s in state],
            "task_goal": [prompt], "is_first_step": self.first})
        self.frames.clear()
        self.first = False
        return np.asarray(result["actions"])

    def close(self):
        self.client._ws.close()


class SimpleMem:
    def __init__(self, paths, enhanced=False):
        if enhanced:
            from .ri_mem.head import load_parameters
            load_parameters(paths["simplemem_ri_head"])
        sys.path.insert(0, str(paths["simplemem_repo"]))
        from robomemarena_sim.policy_adapter import SimpleMemVLARoboMemArenaAdapter
        self.adapter = SimpleMemVLARoboMemArenaAdapter(checkpoint_dir=str(paths["simplemem_checkpoint"]),
                                                      attn_implementation="sdpa")
        self.enhanced, self.binding, self.frame = enhanced, None, -1
        if enhanced:
            from .ri_mem.query import QwenService
            from .ri_mem.engine import RIMem
            self.engine = RIMem(QwenService(str(paths["qwen_checkpoint"]), paths.get("qwen_device", "cuda:0"),
                paths["cache"] / "query_plans"), str(paths["simplemem_ri_head"]), 16384)

    def seed_episode(self, seed):
        self.adapter.set_episode_seed_base(seed)

    def reset(self):
        if self.binding:
            self.binding.close()
            self.binding = None
        if self.enhanced:
            self.engine.close()
        self.frame = -1
        self.adapter.reset()

    def observe(self, obs, prompt, resize_size):
        self.frame += 1
        self.adapter.observe(obs, prompt, resize_size)
        if self.enhanced:
            if self.frame == 0:
                from .ri_mem.bindings import SimpleMemBinding
                self.engine.reset(prompt, obs["observation/image"])
                self.binding = SimpleMemBinding(self.adapter.policy, self.engine)
            self.engine.observe(obs["observation/image"], self.frame)

    def infer_actions(self, obs, prompt, resize_size):
        if self.enhanced:
            self.engine.begin_decision(obs["observation/image"])
        return self.adapter.infer_actions(obs, prompt, resize_size)

    def close(self):
        if self.binding:
            self.binding.close()
        if self.enhanced:
            self.engine.close()


class MemER:
    """Official MemER subgoal predictor with an Arena pi0.5 action policy."""
    def __init__(self, paths, host="127.0.0.1", port=8800):
        import os
        sys.path.insert(0, str(paths["mme_repo"] / "examples/robomme"))
        os.environ["QWEN3VL_PATH"] = str(paths["qwen_checkpoint"])
        from subgoal_prediction.qwenvl.api_memer import Qwen3VLModelMemER
        class ArenaMemER(Qwen3VLModelMemER):
            def _get_current_execution_frame_paths(self):
                n = len(self.execution_frame_paths)
                return [self.execution_frame_paths[i] for i in list(range(n - 1, -1, -2))[:8][::-1]]

            def merge_key_frame_paths(self, dist=8):
                if self.key_frame_paths:
                    super().merge_key_frame_paths(dist)

            def call(self):
                try:
                    subgoal = super().call()
                except (IndexError, json.JSONDecodeError):
                    subgoal = self.subgoals[-1] if self.subgoals else self.task_goal
                self.subgoals.append(subgoal)
                return subgoal
        self.predictor = ArenaMemER(str(paths["memer_adapter"]))
        self.action = Pi05(host, port)
        self.cache, self.episode = paths["cache"] / "memer", 0
        self.started = False

    def reset(self):
        self.episode += 1
        self.started = False
        self.action.reset()

    def observe(self, obs, prompt, resize_size):
        if not self.started:
            self.predictor.start_new_episode(str(self.cache / f"episode_{self.episode}"), None, task_goal=prompt)
            self.started = True
        self.predictor.add_execution_frame(obs["observation/image"])

    def infer_actions(self, obs, prompt, resize_size):
        subgoal = self.predictor.call()
        if not isinstance(subgoal, str) or not subgoal.strip():
            raise RuntimeError("MemER returned an empty subgoal")
        return self.action.infer_actions(obs, subgoal, resize_size)

    def close(self):
        self.action.close()


def generic_adapter(method, paths, backend):
    if method.startswith("simplememvla"):
        return SimpleMem(paths, method.endswith("_ri"))
    if method.startswith("fs_mod"):
        return FrameSample(**backend, expected_method=method)
    if method == "memer":
        return MemER(paths, **backend)
    if method == "pi05":
        return Pi05(**backend)
    raise ValueError(f"{method} uses the native dual-system episode loop")


def native_dual(method, paths, backend, task, env, prompt, output, runtime, specs, logger):
    import eval_fullvlm26_async_vlm_vla as ref
    args = ref.BaseArgs()
    args.resize_size, args.replan_steps, args.num_steps_wait = 256, 10, 10
    args.max_steps, args.post_goal_steps = runtime["max_steps"], runtime["post_goal_steps"]
    args.async_vlm, args.vlm_interval, args.vlm_queue_size = True, 5, 1
    args.n_recent, args.k_max, args.d_merge = 5, 0, 6
    args.vlm_use_wrist, args.vlm_use_keyframe_memory = True, True
    args.vlm_input_profile = "fullvlm_256"
    args.vlm_match_training_jpeg_roundtrip = False
    args.fail_on_extra_pour, args.extra_pour_monitor_steps = True, 30
    infos = ref.load_task_infos(ref.REFERENCE_DIR / "fullvlm_v2_26_memory_tasks.json")
    if method == "predimem":
        checkpoint = paths.get("predimem_task1_planner", paths["predimem_checkpoint"] / "vlm_task1") if task == 1 else paths["predimem_planner"]
        planner = ref.FullVlm26MemoryPlanner(base_model_dir=str(checkpoint),
            lora_path="none", instruction=prompt, system_prompt=ref.SYSTEM_PROMPT_MEMORY,
            prompt_profile="task1_kf5", n_recent=5, d_merge=6, k_max=0, use_keyframe_memory=True,
            max_new_tokens=256, device=backend.get("planner_device", "cuda:0"), logger=logger,
            vlm_model_type=args.vlm_model_type, enable_thinking=False, crop_right_half=False,
            use_wrist=True, task_info=infos[task])
        client = ref.StableWebsocketClientPolicy(backend.get("host", "127.0.0.1"), backend["port"],
                                               ping_interval=None, ping_timeout=None)
    else:
        sys.path.insert(0, str(paths["traceflow_repo"] / "openpi"))
        from optimus_eval.predimem_remote_planner import RemotePrediMemPlanner
        from optimus_eval.predimem_contextual_pi_client import ContextualPiClient
        raw = ref.StableWebsocketClientPolicy(backend.get("host", "127.0.0.1"), backend["lower_port"],
                                             ping_interval=None, ping_timeout=None)
        client = ContextualPiClient(raw, environment_id=f"ri-task{task}-{output.name}")
        client.set_episode_context(task_id=task, episode_idx=0, seed=runtime["seed"])
        planner = RemotePrediMemPlanner(host=backend.get("host", "127.0.0.1"), port=backend["upper_port"],
            environment_id=f"ri-task{task}-{output.name}", task_info=infos[task],
            lower_feature_provider=client.latest_lower_retrieval_feature,
            lower_probe_provider=client.latest_lower_probe_results,
            lower_probe_candidate_consumer=client.set_lower_probe_candidates)
    planner.reset_episode(instruction=prompt, run_dir=output / "planner", logger=logger)
    try:
        result = ref.run_episode_async_stateful(task_id=task, env=env, client=client, planner=planner,
            args=args, logger=logger, stage_specs=specs,
            goal_monitor_dict=ref.ec._build_goal_monitor_dict(ref.ec._resolve_bddl_path(task)),
            goal_check_override=ref._goal_override_check(task), vlm_camera_pose=None,
            fail_on_extra_pour=True, extra_pour_monitor_steps=30,
            post_goal_steps=runtime["post_goal_steps"])
        if getattr(planner, "last_error", None):
            raise RuntimeError(planner.last_error)
        return result
    finally:
        connection = getattr(getattr(client, "client", client), "_ws", None)
        if connection is not None:
            connection.close()
        if hasattr(planner, "close"):
            planner.close()
