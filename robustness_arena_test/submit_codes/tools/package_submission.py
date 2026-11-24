"""Create a source-and-evidence archive, excluding caches and external assets."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=root.parent / "robomemarena_ri_submission.zip")
parser.add_argument("--videos-only", action="store_true")
args = parser.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)
skip = {".cache", ".pytest_cache", "__pycache__", ".ipynb_checkpoints", "runs", "checkpoints"}
source = root / "appendix_videos" if args.videos_only else root
files = sorted(p for p in source.rglob("*") if p.is_file()
               and not any(part in skip or part.endswith(".egg-info") for part in p.relative_to(root).parts)
               and p.suffix not in (".pyc", ".zip"))
with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
    for path in files:
        archive.write(path, Path("submit_codes") / path.relative_to(root))
with zipfile.ZipFile(args.out) as archive:
    assert archive.testzip() is None
    assert sum(n.endswith("/failure.mp4") for n in archive.namelist()) == 10
    assert sum(n.endswith("/clean.mp4") for n in archive.namelist()) == 10
print(json.dumps({"archive": str(args.out.resolve()), "files": len(files), "bytes": args.out.stat().st_size,
                  "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}))
