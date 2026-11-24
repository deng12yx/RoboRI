"""Execute the portable example notebook using this Python environment."""

from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

HERE = Path(__file__).resolve().parents[1]
path = HERE / "notebooks/robomme_ri_demo.ipynb"
nb = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv[0] = sys.executable
client = NotebookClient(
    nb, km=manager, timeout=120, resources={"metadata": {"path": str(path.parent)}}
)
client.execute()
nbformat.write(nb, path)
print(f"Executed {sum(c.cell_type == 'code' for c in nb.cells)} code cells")
