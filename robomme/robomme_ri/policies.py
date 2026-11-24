"""Policies share reset/observe/act; the simulator loop is method independent."""

from pathlib import Path
import numpy as np
from .baselines import METHODS


def frames(obs):
    return [
        (
            np.asarray(f, np.uint8),
            np.asarray(w, np.uint8),
            np.concatenate((np.asarray(j, np.float32), np.asarray(g, np.float32)[:1])),
        )
        for f, w, j, g in zip(
            obs["front_rgb_list"],
            obs["wrist_rgb_list"],
            obs["joint_state_list"],
            obs["gripper_state_list"],
        )
    ]


def instruction(info):
    g = info["task_goal"]
    return str(g[0] if isinstance(g, (list, tuple)) else g)


class MMEPolicy:
    action_space = "joint_angle"

    def __init__(
        self,
        method,
        checkpoint,
        rimem=None,
        subgoals=None,
        retrieval=None,
        control=None,
    ):
        from mme_vla_suite.policies import policy_config
        from mme_vla_suite.training import config

        self.method = METHODS[method]
        self.rimem, self.subgoals, self.retrieval = rimem, subgoals, retrieval
        self.control = control
        self.env_kwargs = (
            {"include_maniskill_obs": True}
            if control and control.kind == "oracle-mem"
            else {}
        )
        self.detach = None
        if (
            self.method.native != "pi05-no-memory"
            and not (Path(checkpoint).parent / "history_config.txt").is_file()
        ):
            raise FileNotFoundError(f"Missing history_config.txt beside {checkpoint}")
        if rimem or control:
            from .rimem.mme import install

            install()
        self.inner = policy_config.create_trained_policy(
            config.get_config("pi05_baseline" if method == "pi05" else "mme_vla_suite"),
            Path(checkpoint),
            seed=7,
        )
        if self.method.native != "pi05-no-memory" and self.inner.config is None:
            raise RuntimeError(
                "Checkpoint loaded without the requested memory configuration"
            )
        if (
            "modul" in self.method.native
            and self.inner._model.integration_type != "modulation"
        ):
            raise ValueError(
                "RI-Mem modulation adapter requires a modulation checkpoint"
            )
        self.metadata = {
            "checkpoint": str(checkpoint),
            "native": self.method.native,
            "subgoals": self.method.subgoals,
        }

    def reset(self, spec, obs, info, env):
        if self.detach:
            self.detach()
        self.inner.reset()
        self.inner._seed = spec.model_seed
        import jax

        self.inner._rng = jax.random.key(spec.model_seed)
        self.goal, self.env = instruction(info), env
        self.count = 0
        if self.control:
            from .history_controls import attach_mme

            self.control.reset(spec.model_seed, env)
            self.detach = attach_mme(self.inner, self.control)
        if self.rimem:
            from .rimem.mme import attach

            self.rimem.reset(self.goal, spec.model_seed)
            self.detach = attach(self.inner, self.rimem, self.method.native)
        if self.subgoals:
            self.subgoals.reset(spec, obs, self.goal)
        self.observe(obs, initial=True)

    def observe(self, obs, initial=False):
        fs = frames(obs)
        if self.control:
            self.control.observe(obs)
        if self.rimem:
            for f, w, s in fs:
                self.rimem.observe(f)
        if self.retrieval and initial:
            # Retrieval comparator changes selection; RI-Mem itself never does.
            ids = self.retrieval([f for f, w, s in fs[:-1]], fs[-1][0])
            fs = [fs[i] for i in ids] + fs[-1:]
        for begin in range(0, len(fs), 32):
            part = fs[begin : begin + 32]
            self.inner.add_buffer(
                dict(
                    images=np.stack([f for f, w, s in part])[:, None],
                    state=np.stack([s for f, w, s in part]),
                    exec_start_idx=len(fs) - 1 if initial and begin == 0 else -1,
                )
            )
        if self.subgoals and not initial:
            self.subgoals.observe(obs, self.count)
        self.current = fs[-1]
        if not initial:
            self.count += len(fs)

    def act(self, obs, info):
        f, w, s = self.current
        inp = {
            "observation/image": f,
            "observation/wrist_image": w,
            "observation/state": s,
            "prompt": self.goal.lower(),
        }
        if self.method.subgoals == "oracle":
            key = (
                "grounded_subgoal"
                if "grounded" in self.method.native
                else "simple_subgoal"
            )
            native_key = key + "_online"
            value = info.get(native_key)
            if isinstance(value, list):
                value = value[-1]
            if not value:
                raise ValueError(f"Oracle variant requires {native_key}")
            inp.update(simple_subgoal=value, grounded_subgoal=value)
        elif self.method.subgoals == "qwen":
            if self.subgoals is None:
                raise ValueError(
                    "Qwen/MemER predictor is required; oracle substitution is forbidden"
                )
            sg = self.subgoals.predict(f, self.count)
            inp.update(simple_subgoal=sg, grounded_subgoal=sg)
        return np.asarray(self.inner.infer(inp)["actions"], np.float32)

    def close(self):
        if self.detach:
            self.detach()
        if self.rimem:
            self.rimem.close()
        if self.subgoals:
            self.subgoals.close()


class SimpleMemPolicy:
    action_space = "joint_angle"

    def __init__(self, checkpoint, rimem=None, control=None):
        import torch
        from transformers import AutoProcessor
        from robomme_sim.policy import get_vla, RoboMMEPolicy
        from robomme_sim.batched_policy import BatchedEvalPolicy

        self.torch, self.rimem = torch, rimem
        self.control = control
        self.control_detach = None
        self.env_kwargs = (
            {"include_maniskill_obs": True}
            if control and control.kind == "oracle-mem"
            else {}
        )
        vla, unnorm, norm = get_vla(
            str(checkpoint), "bfloat16", attn_implementation="sdpa"
        )
        processor = AutoProcessor.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            padding_side="right",
            model_max_length=8192,
        )
        self.buffer = RoboMMEPolicy(
            vla,
            processor,
            unnorm,
            norm,
            num_denoising_steps=10,
            temperature=1.0,
            max_subtask_tokens=64,
        )
        self.batched = BatchedEvalPolicy(
            vla,
            processor,
            unnorm,
            norm,
            num_denoising_steps=10,
            eval_temperature=1.0,
            max_subtask_tokens=64,
        )
        self.norm = norm
        self.bridge = None
        if rimem:
            from .rimem.simplememvla import AttentionBridge

            self.bridge = AttentionBridge(vla)
        self.metadata = {
            "checkpoint": str(checkpoint),
            "backend": "native cached-vision batch-one SDPA",
        }

    def reset(self, spec, obs, info, env):
        if self.control_detach:
            self.control_detach()
        self.buffer.reset()
        self.torch.manual_seed(spec.model_seed)
        self.torch.cuda.manual_seed_all(spec.model_seed)
        self.goal = instruction(info)
        if self.control:
            from .history_controls import attach_simple

            self.control.reset(spec.model_seed, env)
            self.control_detach = attach_simple(self.buffer, self.control)
        if self.rimem:
            self.rimem.reset(self.goal, spec.model_seed)
            self.bridge.clear()
        self.observe(obs)

    def observe(self, obs):
        if self.control:
            self.control.observe(obs)
        for f, w, s in frames(obs):
            self.buffer.observe(
                {"observation.images.front": f, "observation.images.wrist": w}
            )
            if self.rimem:
                self.rimem.observe(f)
            self.state = s

    def act(self, obs, info):
        with self.torch.inference_mode():
            if self.control and self.control.kind == "oracle-mem":
                self.buffer._patch_feat_cache.clear()
            processed = self.buffer._prepare_inputs(self.goal)
            if self.control:
                from .history_controls import simple_processed

                processed = simple_processed(processed, self.control)
            if self.rimem:
                from .rimem.simplememvla import video_keys

                keys, positions = video_keys(processed, self.buffer)
                bias, gate = self.rimem.read(keys)
                self.bridge.set(bias, gate, positions)
            state = (
                self.norm.normalize(self.torch.from_numpy(self.state))
                .unsqueeze(0)
                .to(self.batched.device)
            )
            actions, subtask = self.batched.generate_batch([processed], [state])[0]
        self.last_subtask = subtask
        return np.asarray(actions, np.float32)

    def close(self):
        if self.control_detach:
            self.control_detach()
        if self.bridge:
            self.bridge.close()
        if self.rimem:
            self.rimem.close()


def load_factory(reference, **kwargs):
    """Dependency boundary for official policy runtimes in different venvs."""
    import importlib

    module, name = reference.split(":", 1)
    return getattr(importlib.import_module(module), name)(**kwargs)


def create(method, cfg):
    m = METHODS[method]
    if cfg.get("factory"):
        return load_factory(cfg["factory"], method=method, config=cfg)
    ri = None
    control = None
    if m.history_control:
        from .history_controls import Control

        control = Control(m.history_control)
    if m.rimem:
        from .rimem.runtime import RIMem
        from .rimem.vlm import QwenSCQ

        base = method.removesuffix("+ri-mem")
        scq = QwenSCQ(
            cfg["qwen_model"], cfg.get("qwen_device", "cuda:0"), cfg.get("plan_cache")
        )
        ri = RIMem(base, cfg["head"], scq, cfg.get("ablation", "full"))
    if m.backend == "mme":
        subgoals = None
        if m.subgoals == "qwen":
            from .predictors import create_predictor

            subgoals = create_predictor(m.native, cfg)
        retrieval = None
        if method == "retrieval-fs-mod":
            from .retrieval import SigLIPRetrieval

            retrieval = SigLIPRetrieval(cfg["retrieval_encoder"])
        return MMEPolicy(method, cfg["checkpoint"], ri, subgoals, retrieval, control)
    if m.backend == "simplememvla":
        return SimpleMemPolicy(cfg["checkpoint"], ri, control)
    from .official import create_official

    return create_official(method, cfg)
