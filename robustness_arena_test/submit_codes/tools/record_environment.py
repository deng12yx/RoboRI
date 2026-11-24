from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import argparse
import json
import platform
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
packages = {}
for name in ("numpy", "Pillow", "torch", "transformers", "jax", "jaxlib", "flax", "optax", "mujoco",
             "robosuite", "imageio", "nbformat", "nbclient", "opencv-python", "opencv-python-headless", "h5py", "websockets"):
    try:
        packages[name] = version(name)
    except PackageNotFoundError:
        continue
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps({"python": platform.python_version(), "interpreter": sys.executable,
                               "packages": packages}, indent=2) + "\n")
