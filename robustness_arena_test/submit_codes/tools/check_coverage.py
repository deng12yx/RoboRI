"""Audit the paper method matrix and source-comment language without loading models."""
from collections import Counter
from pathlib import Path
import argparse
import ast
import hashlib
import io
import json
import re
import sys
import tokenize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robo_ri.protocol import CONDITIONS, METHODS, jobs


def audit():
    baselines = {"fs_mod", "fs_mod_retrieval", "simplememvla", "traceflow",
                 "memer", "predimem", "pi05"}
    enhanced = {"fs_mod_ri", "simplememvla_ri"}
    assert set(METHODS) == baselines | enhanced
    assert {name for name, spec in METHODS.items() if spec.get("ri")} == enhanced
    config = json.loads((ROOT / "configs/paper.json").read_text())
    assert set(config["methods"]) == baselines | enhanced
    assert set(config["conditions"]) == {"clean", *CONDITIONS}
    matrix = jobs(config)
    counts = Counter((row["method"], row["condition"]) for row in matrix)
    expected = len(config["tasks"]) * len(config["seeds"])
    assert all(counts[method, condition] == expected
               for method in METHODS for condition in ("clean", *CONDITIONS))

    violations = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part.startswith(".") or part == "__pycache__"
               for part in path.relative_to(ROOT).parts):
            continue
        source = path.read_text()
        comments = [token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
                    if token.type == tokenize.COMMENT]
        comments.extend(ast.get_docstring(node) or "" for node in ast.walk(ast.parse(source))
                        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)))
        if any(re.search(r"[\u3400-\u9fff]", comment) for comment in comments):
            violations.append(str(path.relative_to(ROOT)))
    notebook = json.loads((ROOT / "robomemarena_ri_examples.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown" and re.search(r"[\u3400-\u9fff]", "".join(cell["source"])):
            violations.append(f"robomemarena_ri_examples.ipynb:cell{index}")
    assert not violations, violations
    papers = ROOT.parent / "papers"
    return {
        "check_type": "configuration_and_comment_audit",
        "passed": True,
        "paper_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(papers.glob("Robo_RI_CVPR27*.pdf"))},
        "paper_scope": "Main paper Table 3; supplement sections 2.2 and 2.3",
        "baseline_count": len(baselines),
        "perturbation_count": len(CONDITIONS),
        "baseline_perturbation_combinations": len(baselines) * len(CONDITIONS),
        "ri_methods": sorted(enhanced),
        "excluded_from_arena": {"td_mod_ri": "The paper evaluates TokenDrop+Mod RI-Mem only on RoboMME."},
        "methods": {method: {"clean": counts[method, "clean"],
                             "perturbations": {c: counts[method, c] for c in CONDITIONS}}
                    for method in METHODS},
        "comment_language": "English; no CJK comments, docstrings, or notebook explanations found",
        "full_pretrained_rollouts_verified": False,
        "note": "This audit checks implementation configuration, not measured success or native model execution.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "validation/implementation_coverage.json")
    args = parser.parse_args()
    result = audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("passed", "baseline_count", "perturbation_count",
                       "baseline_perturbation_combinations", "ri_methods")}))
