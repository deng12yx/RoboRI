from __future__ import annotations

import json
from pathlib import Path


SUBMISSION = Path(__file__).resolve().parents[1]


def load_paths(path=None):
    arena = SUBMISSION.parent
    project = arena.parent
    snapshot = arena / "third_party/robomme-mme-vla-all26-eval/runtime"
    defaults = {
        "arena_repo": project / "RoboMemArena", "harness": arena,
        "mme_repo": project / "robomme_policy_learning", "mme_runtime": snapshot,
        "simplemem_repo": arena / "third_party/SimpleMemVLA",
        "traceflow_repo": arena / "third_party/TraceFlow",
        "fs_checkpoint": arena / "checkpoints/mme_vla/hf_ckpt56000",
        "simplemem_checkpoint": arena / "checkpoints/simplememvla_robomemarena",
        "predimem_checkpoint": arena / "checkpoints/PrediMem/e645741c2f34f27e1596bfa89e856d6f3560ed90",
        "cache": SUBMISSION / ".cache",
        "fs_ri_head": SUBMISSION / "checkpoints/fs_mod_ri.npz",
        "simplemem_ri_head": SUBMISSION / "checkpoints/simplememvla_ri.npz",
        "qwen_device": "cuda:0",
    }
    if path:
        base = Path(path).resolve().parent
        for key, value in json.loads(Path(path).read_text()).items():
            if key.endswith("_device"):
                defaults[key] = value
            else:
                candidate = Path(value).expanduser()
                defaults[key] = (base / candidate).resolve() if not candidate.is_absolute() else candidate
    defaults.setdefault("predimem_planner", defaults["predimem_checkpoint"] / "vlm_tasks1to26_ckpt74500")
    defaults.setdefault("pi05_checkpoint", defaults["predimem_checkpoint"] / "vla_alltask")
    return defaults


def activate_fs(paths):
    import os
    import sys
    for path in (paths["mme_repo"] / "packages/openpi-client/src", paths["mme_repo"] / "src",
                 paths["mme_repo"], paths["mme_runtime"] / "src", paths["mme_runtime"]):
        sys.path.insert(0, str(path))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OPENPI_DATA_HOME", str(paths["harness"] / "checkpoints/openpi_data"))
    os.environ.setdefault("PALIGEMMA_TOKENIZER", str(paths["harness"] / "checkpoints/openpi_data/big_vision/paligemma_tokenizer.model"))
    os.chdir(paths["mme_runtime"])
