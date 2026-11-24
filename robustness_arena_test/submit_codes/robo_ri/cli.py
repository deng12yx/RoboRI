from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .paths import SUBMISSION, load_paths
from .protocol import METHODS, CONDITIONS, jobs
from .runner import atomic_json


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--config", type=Path, default=SUBMISSION / "configs/paper.json")
    plan.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--method", choices=list(METHODS))
    run.add_argument("--paths", type=Path)
    run.add_argument("--gpu", default="4")
    run.add_argument("--limit", type=int)
    report = sub.add_parser("report")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    serve = sub.add_parser("serve-fs")
    serve.add_argument("--method", choices=["fs_mod", "fs_mod_retrieval", "fs_mod_ri"], required=True)
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--paths", type=Path)
    serve.add_argument("--gpu", default="4")
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--method", choices=["fs_mod", "simplememvla"], required=True)
    calibrate.add_argument("--samples", type=Path, required=True)
    calibrate.add_argument("--out", type=Path, required=True)
    calibrate.add_argument("--paths", type=Path)
    calibrate.add_argument("--gpu", default="4")
    calibrate.add_argument("--steps", type=int)
    collect = sub.add_parser("collect-calibration")
    collect.add_argument("--method", choices=["fs_mod", "simplememvla"], required=True)
    collect.add_argument("--demonstrations", type=Path, required=True)
    collect.add_argument("--out", type=Path, required=True)
    collect.add_argument("--paths", type=Path)
    collect.add_argument("--gpu", default="4")
    args = parser.parse_args()
    if hasattr(args, "gpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ["MUJOCO_EGL_DEVICE_ID"] = args.gpu.split(",")[0]
        os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.command == "plan":
        config = json.loads(args.config.read_text())
        config["provenance"] = {"source_sha256": {
            str(path.relative_to(SUBMISSION)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((SUBMISSION / "robo_ri").rglob("*.py"))},
            "paper_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((SUBMISSION.parent / "papers").glob("Robo_RI_CVPR27*.pdf"))}}
        rows = jobs(config)
        target = args.out / "manifest.json"
        if target.exists():
            raise FileExistsError(f"Manifest already exists: {target}")
        atomic_json(target, {"schema": 1, "config": config, "jobs": rows})
        print(json.dumps({"manifest": str(target.resolve()), "episodes": len(rows),
                          "perturbations": len(CONDITIONS), "methods": len(config["methods"])}))
    elif args.command == "run":
        from .runner import run_manifest
        run_manifest(args.manifest, load_paths(args.paths), args.method, args.limit)
    elif args.command == "report":
        from .analysis import summarize, paired_success_times
        atomic_json(args.out, {"success_rates": summarize(args.manifest),
                               "paired_success_times": paired_success_times(args.manifest)})
        print(args.out.resolve())
    elif args.command == "serve-fs":
        from .server import serve_fs
        serve_fs(load_paths(args.paths), args.method, args.port)
    elif args.command == "collect-calibration":
        from .collect import collect
        result = collect(args.demonstrations, args.out, load_paths(args.paths), args.method)
        print(result)
    elif args.command == "calibrate":
        from .calibration import calibration_samples, calibrate_fs, calibrate_simplemem
        paths = load_paths(args.paths)
        samples, output = calibration_samples(args.samples), args.out.resolve()
        if args.method == "simplememvla":
            from .arena import activate
            activate(paths["arena_repo"], paths["cache"])
            from .baselines import SimpleMem
            adapter = SimpleMem(paths)
            result = calibrate_simplemem(adapter.adapter.policy.vla, samples, output, args.steps or 16)
        else:
            from .paths import activate_fs
            activate_fs(paths)
            from .ri_mem.bindings import install_fs_mod_readout
            install_fs_mod_readout()
            from mme_vla_suite.policies.policy_config import create_trained_policy
            from mme_vla_suite.training.config import get_config
            checkpoint = paths["cache"] / "fs_checkpoint/56000"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if not checkpoint.exists():
                checkpoint.symlink_to(paths["fs_checkpoint"], target_is_directory=True)
            (checkpoint.parent / "history_config.txt").write_text("perceptual-framesamp-modul.yaml")
            policy = create_trained_policy(get_config("mme_vla_ftv2_all26_framesamp_modul"), checkpoint, seed=7)
            result = calibrate_fs(policy._model, samples, output, args.steps or 32)
        atomic_json(output.with_suffix(".training.json"), result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
