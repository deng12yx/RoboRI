from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from robo_ri.protocol import CONDITIONS

gallery = json.loads((root / "appendix_videos/manifest.json").read_text())
assert {c["condition"] for c in gallery["cases"]} == set(CONDITIONS)
assert len(gallery["cases"]) == 10
video_count = 0
for case in gallery["cases"]:
    directory = root / "appendix_videos" / Path(case["failure_video"]).parent
    records = [json.loads((directory / f"{role}_result.json").read_text()) for role in ("clean", "failure")]
    assert all(row["status"] == "completed" for row in records)
    assert [r["tsr"] for r in records] == [1, 0]
    assert len({(r["task"], r["seed"], r["method"], r["init_state_sha1"]) for r in records}) == 1
    for role in ("clean", "failure"):
        path = root / "appendix_videos" / case[f"{role}_video"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case[f"{role}_sha256"]
        import cv2
        capture = cv2.VideoCapture(str(path))
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.ndim == 3 and frame.shape[-1] == 3
            count += 1
        capture.release()
        assert count == case["video_frames"][role], (path, count)
        video_count += 1
notebook = json.loads((root / "robomemarena_ri_examples.ipynb").read_text())
code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
assert all(c["execution_count"] is not None for c in code_cells)
assert not any(o["output_type"] == "error" for c in code_cells for o in c["outputs"])
assert sum(bool({"image/png", "image/jpeg"} & o.get("data", {}).keys())
           for c in code_cells for o in c["outputs"]) == 10
assert not any(ord(ch) < 32 and ch not in "\n\r\t" for c in notebook["cells"] for ch in "".join(c["source"]))
result = {"matched_video_pairs": 10, "fully_decoded_videos": video_count,
          "executed_notebook_code_cells": len(code_cells), "embedded_comparisons": 10, "passed": True}
(root / "validation/artifacts.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
