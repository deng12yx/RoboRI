"""Generate and optionally run the 11 RI-Mem ablation configurations."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robomme_ri.baselines import ABLATIONS
from robomme_ri.cli import load_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    source = load_configuration(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    bases = ["framesamp-modul", "tokendrop-modul", "simplememvla"]
    for ablation in ABLATIONS:
        cfg = copy.deepcopy(source)
        cfg["episodes"] = list(range(30))
        cfg["methods"] = (
            bases if ablation == "original" else [m + "+ri-mem" for m in bases]
        )
        for method in cfg["methods"]:
            if method.endswith("+ri-mem"):
                cfg["policies"][method]["ablation"] = ablation
        path = args.output / (ablation + ".json")
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        if args.run:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_experiment.py"),
                    str(path),
                    "--output",
                    str(args.output / ablation),
                    "--gpus",
                    args.gpus,
                ],
                check=True,
            )
        print(path)


if __name__ == "__main__":
    main()
