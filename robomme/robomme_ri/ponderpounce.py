"""Run the official PonderPounce callbacks at the native 20 Hz control rate."""

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from .policies import frames, instruction


class PonderPouncePolicy:
    action_space = "joint_angle"
    exec_horizon = 1

    def __init__(self, cfg):
        if cfg.get("source_root"):
            source = Path(cfg["source_root"]).expanduser().resolve()
            if not (source / "ponderpounce/eval/robomme_server.py").is_file():
                raise FileNotFoundError(source / "ponderpounce/eval/robomme_server.py")
            sys.path.insert(0, str(source))
        from ponderpounce.eval.robomme_server import PonderPounceRoboMMEServer

        self.server = PonderPounceRoboMMEServer(
            checkpoint_path=cfg["checkpoint"],
            device=cfg.get("device", "cuda:0"),
            dtype="bfloat16",
            sim_freq_hz=20.0,
        )
        self.loop = asyncio.new_event_loop()
        self.context = None
        self.metadata = {
            "native": "PonderPounceRoboMMEServer",
            "upstream_revision": "3d0ae369568e353a100a688c1f77e9d29c7f06d0",
            "checkpoint": cfg["checkpoint"],
            "control_hz": 20,
            "exec_horizon": 1,
        }

    def reset(self, spec, obs, info, env):
        self._end_episode()
        self.goal = instruction(info)
        # Session identity excludes the condition, preserving paired noise seeds.
        sid = f"robomme:{spec.task}:{spec.episode}"
        self.server._seed = spec.model_seed
        self.server._episode_counts.pop(sid, None)

        async def send_action(action):
            self.latest_action = action

        self.context = SimpleNamespace(session_id=sid, send_action=send_action)
        self.loop.run_until_complete(
            self.server.on_episode_start(
                {
                    "task_description": self.goal,
                    "demo_frames": [f for f, _, _ in frames(obs)[:-1]],
                },
                self.context,
            )
        )
        self.observe(obs)

    def observe(self, obs):
        self.current = frames(obs)[-1]

    def act(self, obs, info):
        front, wrist, state = self.current
        self.latest_action = None
        self.loop.run_until_complete(
            self.server.on_observation(
                {
                    "images": {"agentview": front, "wrist": wrist},
                    "states": state,
                    "task_description": self.goal,
                },
                self.context,
            )
        )
        if self.latest_action is None:
            raise RuntimeError("PonderPounce did not return a control-step action")
        action = np.asarray(self.latest_action["actions"], dtype=np.float32)
        if action.shape != (1, 8):
            raise ValueError(
                f"Expected one RoboMME joint action, received {action.shape}"
            )
        return action

    def _end_episode(self):
        if self.context is not None:
            self.loop.run_until_complete(self.server.on_episode_end({}, self.context))
            self.context = None

    def close(self):
        try:
            self._end_episode()
        finally:
            self.loop.close()
