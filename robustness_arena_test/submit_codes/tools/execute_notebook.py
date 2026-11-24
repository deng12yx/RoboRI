from pathlib import Path
import json
import os

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
kernel = root / ".cache/jupyter/kernels/robo-ri-eval"
kernel.mkdir(parents=True, exist_ok=True)
(kernel / "kernel.json").write_text(json.dumps({
    "argv": [str(root.parent / ".venv-eval/bin/python"), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "RoboMemArena RI", "language": "python"}))
os.environ["JUPYTER_PATH"] = str(root / ".cache/jupyter")
os.environ["PYTHONPATH"] = str(root)
path = root / "robomemarena_ri_examples.ipynb"
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=120, kernel_name="robo-ri-eval",
               resources={"metadata": {"path": str(root)}}).execute()
nbformat.write(notebook, path)
print(f"Executed {len(notebook.cells)} notebook cells")
