"""Run each baseline in its configured Python environment, with paired shards."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robomme_ri.cli import load_configuration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--output", required=True)
    p.add_argument("--gpus", default="0")
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    cfg = load_configuration(a.config)
    output = Path(a.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    gpus = a.gpus.split(",")
    queue = list(cfg["methods"])
    running = []
    failures = []

    def interrupted(*args):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        while queue or running:
            busy = {r["gpu"] for r in running}
            for gpu in gpus:
                if gpu in busy or not queue:
                    continue
                method = queue.pop(0)
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts/run_method.py"),
                    str(Path(a.config).resolve()),
                    "--method",
                    method,
                    "--output",
                    str(output / method),
                    "--gpu",
                    gpu,
                ]
                if a.plan_only:
                    cmd.append("--plan-only")
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=gpu,
                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                    PYTHONPATH=str(ROOT)
                    + os.pathsep
                    + os.environ.get("PYTHONPATH", ""),
                    MUJOCO_GL="egl",
                )
                log = (output / (method + ".log")).open("w")
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running.append(dict(method=method, gpu=gpu, proc=proc, log=log))
            next_running = []
            for r in running:
                code = r["proc"].poll()
                if code is None:
                    next_running.append(r)
                else:
                    r["log"].close()
                    if code:
                        failures.append({"method": r["method"], "returncode": code})
                    print(r["method"], "exit", code, flush=True)
            running = next_running
            if running:
                time.sleep(1)
    finally:
        for r in running:
            if r["proc"].poll() is None:
                os.killpg(r["proc"].pid, signal.SIGTERM)
            r["log"].close()
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
