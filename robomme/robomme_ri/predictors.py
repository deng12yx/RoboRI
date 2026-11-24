"""Baseline subgoal predictors, distinct from RI-Mem's frozen base-model SCQ."""

import tempfile


class NativeQwen:
    def __init__(self, native, cfg):
        import os

        if cfg.get("qwen_model"):
            os.environ["QWEN3VL_PATH"] = cfg["qwen_model"]
        self.memer = native == "memer"
        self.kind = "simple_subgoal" if "simple" in native else "grounded_subgoal"
        if self.memer:
            from subgoal_prediction.qwenvl.api_memer import Qwen3VLModelMemER

            self.api = Qwen3VLModelMemER(adapter_path=cfg["subgoal_adapter"])
        else:
            from subgoal_prediction.qwenvl.api import Qwen3VLModel

            self.api = Qwen3VLModel(
                adapter_path=cfg["subgoal_adapter"], subgoal_type=self.kind
            )
        self.directory = None

    def reset(self, spec, obs, goal):
        self.close()
        self.task = spec.task
        self.last = None
        self.directory = tempfile.TemporaryDirectory(prefix="robomme-ri-qwen-")
        self.api.start_new_episode(
            self.directory.name, obs["front_rgb_list"][:-1], goal
        )
        if self.memer:
            self.api.add_execution_frame(obs["front_rgb_list"][-1])

    def observe(self, obs, step):
        if self.memer:
            for f in obs["front_rgb_list"]:
                self.api.add_execution_frame(f)

    def predict(self, current, step):
        keep = 90 if self.task in ("ButtonUnmask", "PickHighlight") else 0
        if self.task == "ButtonUnmaskSwap" and self.last:
            keep = (
                100
                if "press the first button" in self.last
                else 250
                if "press the second button" in self.last
                else 0
            )
        self.last = (
            self.api.call() if self.memer else self.api.call(current, step, keep)
        )
        if not isinstance(self.last, str) or not self.last.strip():
            raise RuntimeError("Baseline Qwen returned no subgoal")
        return self.last

    def close(self):
        if self.directory:
            self.directory.cleanup()
            self.directory = None


class RemoteQwen:
    def __init__(self, native, cfg):
        from .official import WebSocket

        self.ws = WebSocket(cfg["subgoal_endpoint"])
        if self.ws.metadata.get("native") != native:
            raise ValueError("Subgoal service does not match the requested baseline")
        self.memer = native == "memer"

    def reset(self, spec, obs, goal):
        self.ws.call(
            {
                "op": "reset",
                "spec": spec.as_dict(),
                "front": obs["front_rgb_list"],
                "goal": goal,
            }
        )

    def observe(self, obs, step):
        if self.memer:
            self.ws.call(
                {"op": "observe", "front": obs["front_rgb_list"], "step": step}
            )

    def predict(self, current, step):
        return self.ws.call({"op": "predict", "front": current, "step": step})[
            "subgoal"
        ]

    def close(self):
        self.ws.close()


def create_predictor(native, cfg):
    return (
        RemoteQwen(native, cfg)
        if cfg.get("subgoal_endpoint")
        else NativeQwen(native, cfg)
    )
