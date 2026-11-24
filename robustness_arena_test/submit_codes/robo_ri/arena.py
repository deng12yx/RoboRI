from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def activate(root: str | Path, cache: str | Path):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    benchmark = root / "evaluation_benchmark"
    paths = [benchmark / "scripts", benchmark / "async_vlm26_reference", benchmark / "libero_fork",
             benchmark / "openpi_minimal_runtime", root / "third_party/openpi_minimal/packages/openpi-client/src"]
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    config = cache / "libero"
    config.mkdir(parents=True, exist_ok=True)
    libero = benchmark / "libero_fork/libero"
    (config / "config.yaml").write_text(json.dumps({"benchmark_root": str(libero),
        "bddl_files": str(libero / "bddl_files"), "init_states": str(libero / "init_files"),
        "assets": str(libero / "assets"), "datasets": str(cache / "datasets_unused")}))
    os.environ.update(LIBERO_CONFIG_PATH=str(config), TARGET_LIBERO_PATH=str(benchmark / "libero_fork"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["NO_PROXY"] = ",".join(["127.0.0.1", "localhost", "::1", os.environ.get("NO_PROXY", "")])
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    import eval_common as ec
    import eval_tasks2_26 as evaluation
    evaluation._patch_env_resolution()
    return ec, evaluation
