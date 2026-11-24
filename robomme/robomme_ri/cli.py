"""Evaluation, calibration, inventory and reporting commands."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from .baselines import METHODS, ABLATIONS
from .protocol import CONDITIONS, TASKS, EpisodeSpec


def load_configuration(path):
    import os

    root = Path(__file__).resolve().parents[3]
    defaults = {
        "LAPSE_ROOT": str(root),
        "SUBMIT_ROOT": str(Path(__file__).resolve().parents[1]),
        "MODEL_ROOT": "/path/to/mme_vla_suite",
        "QWEN_MODEL": "/path/to/Qwen3-VL-4B-Instruct",
    }
    text = Path(path).read_text()
    for name, value in defaults.items():
        text = text.replace("${" + name + "}", os.environ.get(name, value))
    data = json.loads(os.path.expandvars(text))
    if data.get("benchmark") != "robomme":
        raise ValueError("This submission supports RoboMME only")
    return data


def grid(cfg):
    for method in cfg["methods"]:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method}")
        for task in cfg.get("tasks", TASKS):
            for ep in cfg.get("episodes", range(50)):
                for condition in cfg.get("conditions", ["clean", *CONDITIONS]):
                    pcfg = cfg.get("policies", {}).get(method, {})
                    yield EpisodeSpec(
                        method,
                        task,
                        ep,
                        condition,
                        model_seed=pcfg.get("model_seed", cfg.get("model_seed", 7)),
                        max_steps=cfg.get("max_steps", 1300),
                        exec_horizon=pcfg.get(
                            "exec_horizon", cfg.get("exec_horizon", 16)
                        ),
                    )


def add_source_paths(cfg):
    for p in reversed(cfg.get("python_paths", [])):
        path = Path(p).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        sys.path.insert(0, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    for cmd in ("plan", "evaluate"):
        p = sub.add_parser(cmd)
        p.add_argument("config")
        p.add_argument("--output", required=True)
        p.add_argument("--method", choices=METHODS)
        p.add_argument("--shard", default="0/1")
    p = sub.add_parser("summarize")
    p.add_argument("results")
    p.add_argument("--output", required=True)
    p = sub.add_parser("calibrate")
    p.add_argument("config")
    p.add_argument(
        "--backbone",
        choices=("framesamp-modul", "tokendrop-modul", "simplememvla"),
        required=True,
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        print(
            json.dumps(
                {
                    "conditions": {k: asdict(v) for k, v in CONDITIONS.items()},
                    "methods": {k: asdict(v) for k, v in METHODS.items()},
                    "ablations": ABLATIONS,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "summarize":
        from .evaluate import summarize

        rows = [
            json.loads(l)
            for l in Path(args.results).read_text().splitlines()
            if l.strip()
        ]
        Path(args.output).write_text(
            json.dumps(summarize(rows), indent=2, ensure_ascii=False) + "\n"
        )
        return
    cfg = load_configuration(args.config)
    if args.command == "calibrate":
        add_source_paths(cfg)
        from .calibrate import train_mme, train_simplememvla

        ckpt = cfg["policies"][args.backbone]["checkpoint"]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if args.backbone == "simplememvla":
            train_simplememvla(ckpt, args.manifest, args.output)
        else:
            train_mme(ckpt, args.manifest, args.output, args.backbone)
        return
    i, n = map(int, args.shard.split("/"))
    if not 0 <= i < n:
        parser.error("Require 0 <= shard index < shard count")
    specs = [s for s in grid(cfg) if (not args.method or s.method == args.method)]
    # Paired conditions always remain on the same task/episode shard.
    pairs = list(dict.fromkeys((s.task, s.episode) for s in specs))
    selected = set(pairs[i::n])
    specs = [s for s in specs if (s.task, s.episode) in selected]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    (out / f"plan_{i}.json").write_text(
        json.dumps(
            {
                "config_hash": sha,
                "config": cfg,
                "episodes": [s.as_dict() for s in specs],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"{len(specs)} episodes; clean + {len(CONDITIONS)} independent perturbations; config {sha[:12]}"
    )
    if args.command == "plan":
        return
    add_source_paths(cfg)
    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from .evaluate import run_episode
    from .policies import create

    target = out / f"shard_{i}"
    done = set()
    if (target / "rollouts.jsonl").exists():
        for l in (target / "rollouts.jsonl").read_text().splitlines():
            r = json.loads(l)
            if r.get("config_hash") == sha and r["status"] != "technical_error":
                done.add(r["key"])
    policy = method = None
    try:
        for spec in specs:
            if spec.key() in done:
                continue
            if method != spec.method:
                if policy:
                    policy.close()
                method = spec.method
                pcfg = cfg["policies"][method]
                policy = create(method, pcfg)
            r = run_episode(spec, policy, target, cfg.get("save_video", False), sha)
            print(
                json.dumps(
                    {
                        k: r.get(k)
                        for k in (
                            "method",
                            "task",
                            "episode",
                            "condition",
                            "status",
                            "event_state",
                        )
                    }
                ),
                flush=True,
            )
    finally:
        if policy:
            policy.close()


if __name__ == "__main__":
    main()
