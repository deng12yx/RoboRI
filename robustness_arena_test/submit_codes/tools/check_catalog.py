"""Validate all 26 target profiles against native BDDL and stage definitions."""
from pathlib import Path
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robo_ri.paths import load_paths
from robo_ri.perturbations import TARGETS
from robo_ri.arena import activate

paths = load_paths()
ec, generic = activate(paths["arena_repo"], paths["cache"])
rows = []
for task, targets in TARGETS.items():
    bddl = ec._resolve_bddl_path(task)
    text = bddl.read_text()
    for target in targets:
        if not re.search(rf"\b{re.escape(target)}\s+-", text):
            raise ValueError(f"Unknown target {target} in task {task}")
    stages = [stage.name for stage in generic._task_specs(task)]
    if not stages:
        raise ValueError(f"No native stages for task {task}")
    rows.append({"task": task, "targets": targets, "stages": stages, "bddl": bddl.name})
target = Path(__file__).resolve().parents[1] / "validation/catalog.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(rows, indent=2) + "\n")
print(f"Validated {len(rows)} native task profiles")
