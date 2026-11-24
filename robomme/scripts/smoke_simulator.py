"""Exercise all event types with a stationary controller; no TSR is reported."""

import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
from robomme_ri.protocol import CONDITIONS, EpisodeSpec
from robomme_ri.evaluate import run_episode


class Stationary:
    action_space = "joint_angle"
    evaluation_kind = "interface_smoke"
    metadata = {"test_controller": "stationary; not a baseline"}

    def __init__(self, oracle_masks=False):
        self.oracle = oracle_masks
        self.env_kwargs = {"include_maniskill_obs": True} if oracle_masks else {}
        self.metadata = {
            "test_controller": "stationary; not a baseline",
            "decision_inputs": [],
        }
        self.observed = 0

    def reset(self, spec, obs, info, env):
        if self.oracle:
            from robomme_ri.history_controls import Control

            self.control = Control("oracle-mem")
            self.control.reset(spec.model_seed, env)
        self.observe(obs)

    def observe(self, obs):
        if self.oracle:
            self.control.observe(obs)
        self.joints = np.asarray(obs["joint_state_list"][-1]).reshape(-1)[:7]
        import hashlib

        self.observed += len(obs["front_rgb_list"])
        self.image_hash = hashlib.sha256(
            np.asarray(obs["front_rgb_list"][-1], np.uint8).tobytes()
        ).hexdigest()

    def act(self, obs, info):
        self.metadata["decision_inputs"].append(
            {"observed_frames": self.observed, "front_sha256": self.image_hash}
        )
        return np.tile(np.r_[self.joints, 1.0].astype(np.float32), (16, 1))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=["clean", *CONDITIONS])
    parser.add_argument("--oracle-masks", action="store_true")
    args = parser.parse_args()
    output = ROOT / "outputs/simulator_smoke"
    for condition in args.conditions:
        spec = EpisodeSpec("stationary-test", "PickXtimes", 2, condition, max_steps=110)
        row = run_episode(spec, Stationary(args.oracle_masks), output, save_video=True)
        print(
            condition,
            row["status"],
            row["event_state"],
            row.get("error", ""),
            flush=True,
        )
        if row["status"] == "technical_error":
            raise SystemExit(1)
