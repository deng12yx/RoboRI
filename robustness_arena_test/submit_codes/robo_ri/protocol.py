from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    start: int
    end: int
    variants: tuple[str, ...] = ("default",)


CONDITIONS = {
    c.name: c for c in (
        Condition("onset_arrival", "onset_arrival", 10, 50),
        Condition("mid_arrival", "mid_arrival", 10, 120),
        Condition("mid_shift", "displacement", 80, 2500),
        Condition("remove_return", "displacement", 80, 160),
        Condition("tv_on", "distractor", 40, 280),
        Condition("object_passby", "distractor", 40, 280),
        Condition("unrelated_object", "distractor", 40, 280, ("insert", "remove")),
        Condition("target_like_object", "distractor", 40, 280, ("insert", "remove")),
        Condition("camera_shake", "distractor", 40, 43),
        Condition("light_change", "distractor", 40, 43, ("bright", "dark")),
    )
}

METHODS = {
    "fs_mod": {"label": "FS+Mod (MME-VLA)", "chunk": 20},
    "simplememvla": {"label": "SimpleMemVLA", "chunk": 10},
    "traceflow": {"label": "TraceFlow", "chunk": 10},
    "memer": {"label": "MemER", "chunk": 10},
    "predimem": {"label": "PrediMem", "chunk": 10},
    "fs_mod_retrieval": {"label": "FS+Mod (Retrieval)", "chunk": 20},
    "pi05": {"label": "pi0.5", "chunk": 10},
    "fs_mod_ri": {"label": "FS+Mod + RI-Mem", "chunk": 20, "ri": True},
    "simplememvla_ri": {"label": "SimpleMemVLA + RI-Mem", "chunk": 10, "ri": True},
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def variant(condition: str, seed: int) -> str:
    if condition == "clean":
        return "default"
    choices = CONDITIONS[condition].variants
    return choices[seed % len(choices)]


def jobs(config: dict) -> list[dict]:
    names = config["conditions"]
    if len(names) != len(set(names)) or set(names) - {"clean", *CONDITIONS}:
        raise ValueError("Conditions must be unique canonical names; late variants are unsupported")
    if len(config["methods"]) != len(set(config["methods"])):
        raise ValueError("Duplicate method")
    if set(config["methods"]) - METHODS.keys():
        raise ValueError("Unknown baseline")
    tasks, seeds = config["tasks"], config["seeds"]
    if not tasks or not seeds or len(set(tasks)) != len(tasks) or len(set(seeds)) != len(seeds):
        raise ValueError("Tasks and seeds must be nonempty and unique")
    if set(tasks) - set(range(1, 27)):
        raise ValueError("RoboMemArena task IDs are 1–26")
    protocol = {"conditions": {k: asdict(v) for k, v in CONDITIONS.items()},
                "runtime": config["runtime"], "version": 1, "provenance": config.get("provenance", {})}
    protocol_hash = digest(protocol)
    return [{"id": f"{m}__t{t:02d}_s{s}__{c}", "method": m, "task": t, "seed": s,
             "condition": c, "variant": variant(c, s), "protocol_hash": protocol_hash}
            for m in config["methods"] for t in tasks for s in seeds for c in names]
