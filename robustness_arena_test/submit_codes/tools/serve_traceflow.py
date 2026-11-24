"""Start released TraceFlow upper/lower services without launching its environment workers."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robo_ri.paths import SUBMISSION, load_paths

parser = argparse.ArgumentParser()
parser.add_argument("--suite", choices=["sequence", "transferring", "counting", "occlusion"], required=True)
parser.add_argument("--upper-gpu", default="4")
parser.add_argument("--lower-gpu", default="7")
parser.add_argument("--upper-port", default="8830")
parser.add_argument("--lower-port", default="8831")
parser.add_argument("--paths", type=Path)
parser.add_argument("--prepare-only", action="store_true")
args = parser.parse_args()
paths = load_paths(args.paths)
arena, tf = paths["harness"], paths["traceflow_repo"]
root = tf / "openpi"
source = root / "scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh"
text = source.read_text()
anchor = "  worker_pids=()\n  video_arg=--save-video"
if text.count(anchor) != 1:
    raise RuntimeError("Unsupported TraceFlow launcher revision")
text = text.replace(anchor, '  echo "TraceFlow services ready; environment workers are supplied by robo_ri."\n'
                    '  wait "${lower_pid}"\n  exit $?\n' + anchor, 1)
directory = paths["cache"] / f"traceflow_{args.suite}"
directory.mkdir(parents=True, exist_ok=True)
snapshot = directory / "serve_native.sh"
snapshot.write_text(text)
profiles = {
    "sequence": ("extra8", "upper", "v0", "1,2,3,22"),
    "transferring": ("extra8", "fusion", "v1", "18,19,25,26"),
    "counting": ("counting", "fusion", "v1", "6,7,8,9,10,15,16"),
    "occlusion": ("occlusion", "fusion", "v1", "4,5,11,12,13,14,17,20,21,23,24"),
}
group, head, mode, tasks = profiles[args.suite]
assets = paths.get("traceflow_assets", arena / "checkpoints/traceflow_models") / "arena" / group
checkpoint = paths.get("traceflow_action_checkpoint", arena / "checkpoints/traceflow_official/predimem/checkpoints/vla_alltask_pytorch")
env = {**os.environ, "ROOT": str(root), "PREDIMEM_PROJECT_ROOT": str(root), "PREDIMEM_DUAL_RUN_SNAPSHOT": "1",
    "ARENA_ROOT": str(paths["arena_repo"]), "AOSS_ROOT": str(assets), "HEAD_VARIANTS": head,
    "OPENPI_PY": str(arena / ".venv-tf-openpi/bin/python"), "VLM_PY": str(arena / ".venv-tf-predimem/bin/python"),
    "VLA_CKPT": str(checkpoint), "VLM_CKPT": str(paths["predimem_planner"]),
    "OPENPI_DATA_HOME": str(arena / "checkpoints/openpi_data"), "RUN_ROOT": str(directory / "logs"),
    "TASK_IDS": tasks, "MEMORY_ALLOWED_TASK_IDS": tasks, "HEAD_VARIANTS": head,
    "UPPER_GPU": args.upper_gpu, "LOWER_GPU": args.lower_gpu,
    "UPPER_PORT": args.upper_port, "LOWER_PORT": args.lower_port,
    "MEMORY_TOP_K": "16", "NEGATIVE_MEMORY_TOP_K": "8", "MEMORY_EXACT_TASK_GATE": "1",
    "MEMORY_GUIDANCE_NORM_CAP": "0.20", "MEMORY_GUIDANCE_TOTAL_NORM_CAP": "0.20",
    "MEMORY_ALIGNMENT_TAG": "joint-tdense5-stage-frame-v3-upper-hold10", "ALLOW_RETRIEVAL_HEAD_PATH_REUSE": "1",
    "ENV_WORKERS": "1", "UPPER_BATCH_SIZE": "1", "LOWER_BATCH_SIZE": "1", "SEED": "50",
    "PYTHONPATH": os.pathsep.join(map(str, [tf / "src", root / "src", root / "packages/openpi-client/src", root])),
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "NO_PROXY": "127.0.0.1,localhost,::1", "no_proxy": "127.0.0.1,localhost,::1"}
if args.suite in ("counting", "occlusion"):
    env["MEMORY_GUIDANCE_SUITE_GATE_PATH"] = str(root / f"configs/robomemarena/arena_{args.suite}_v1_1_gate.json")
subprocess.run(["bash", "-n", str(snapshot)], check=True)
if args.prepare_only:
    print(snapshot)
else:
    os.execvpe("bash", ["bash", str(snapshot), mode], env)
