"""Main paper §3, §5.1; supplement Table 2. All times are 20 Hz steps."""

from dataclasses import dataclass, asdict
import hashlib
import json

TASKS = (
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)


@dataclass(frozen=True)
class Condition:
    name: str
    group: str
    start: int
    end: int | None
    parameters: dict
    clock: str = "online"


CONDITIONS = {
    "onset_arrival": Condition(
        "Onset Arrival", "remember", 0, 20, dict(offset_m=0.65, height_m=0.20), "native"
    ),
    "mid_arrival": Condition(
        "Mid-task Arrival", "remember", 0, 12, dict(offset_m=0.65, height_m=0.25)
    ),
    "shift": Condition(
        "Slight Shift", "remember", 40, None, dict(distance_m=0.02, directions=4)
    ),
    "remove_return": Condition(
        "Remove & Return", "remember", 40, 52, dict(offset_m=[0.0, 0.65, 0.25])
    ),
    "tv": Condition("TV On", "ignore", 16, 96, dict(blocks=4, update_steps=8)),
    "passby": Condition("Object Pass-by", "ignore", 32, 96, dict(path_m=0.68)),
    "unrelated": Condition(
        "Unrelated Object",
        "ignore",
        16,
        None,
        dict(remove_step=48, subtypes=["insert", "remove"]),
    ),
    "target_like": Condition(
        "Target-like Object",
        "ignore",
        16,
        None,
        dict(
            remove_step=48, size_m=0.05, white_cross=True, subtypes=["insert", "remove"]
        ),
    ),
    "camera": Condition("Camera Shake", "ignore", 16, 19, dict(amplitude_m=0.018)),
    "light": Condition(
        "Light Change",
        "ignore",
        16,
        19,
        dict(bright=[2.0, 2.0, 2.0], dark=[0.0, 0.0, 0.0], subtypes=["bright", "dark"]),
    ),
}


@dataclass(frozen=True)
class EpisodeSpec:
    method: str
    task: str
    episode: int
    condition: str = "clean"
    subtype: str | None = None
    model_seed: int = 7
    max_steps: int = 1300
    exec_horizon: int = 16

    def __post_init__(self):
        if self.task not in TASKS or self.condition not in ("clean", *CONDITIONS):
            raise ValueError(f"Unknown task/condition: {self.task}/{self.condition}")
        if not 0 <= self.episode < 50 or self.max_steps <= 0 or self.exec_horizon <= 0:
            raise ValueError("Use the 50 official test episodes and positive budgets")
        allowed = (
            CONDITIONS[self.condition].parameters.get("subtypes", [])
            if self.condition != "clean"
            else []
        )
        if self.subtype is not None and self.subtype not in allowed:
            raise ValueError(f"Invalid subtype {self.subtype}")

    @property
    def resolved_subtype(self):
        """Balanced assignment, independent of policy; explicit subtype overrides it."""
        choices = (
            CONDITIONS[self.condition].parameters.get("subtypes", [])
            if self.condition != "clean"
            else []
        )
        return self.subtype or (
            choices[self.episode % len(choices)] if choices else None
        )

    def event_seed(self, official_seed):
        payload = f"robomme-ri/v1/{self.task}/{official_seed}/{self.condition}/{self.resolved_subtype}"
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")

    def as_dict(self):
        return {**asdict(self), "subtype": self.resolved_subtype}

    def key(self):
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode()
        ).hexdigest()[:20]
