from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean


def summarize(manifest):
    manifest = Path(manifest)
    data = json.loads(manifest.read_text())
    groups = defaultdict(list)
    for job in data["jobs"]:
        path = manifest.parent / "episodes" / job["id"] / "result.json"
        row = json.loads(path.read_text()) if path.exists() else None
        if row and (row.get("id") != job["id"] or row.get("protocol_hash") != job["protocol_hash"]):
            raise ValueError(f"Unmatched result: {path}")
        groups[job["method"], job["condition"]].append((job, row))
    report = []
    for (method, condition), entries in sorted(groups.items()):
        completed = [(j, r) for j, r in entries if r and r["status"] == "completed"]
        valid = [(j, r) for j, r in completed if r.get("event", {}).get("valid")]
        defaults = [(j, r) for j, r in valid if r["event"].get("paper_default_timing_met", True)]
        def macro(rows):
            by_task = defaultdict(list)
            for job, result in rows:
                by_task[job["task"]].append(float(result["success"]))
            return 100 * mean(mean(s) for s in by_task.values()) if by_task else None
        report.append({"method": method, "condition": condition, "planned": len(entries),
            "completed": len(completed), "valid_events": len(valid),
            "paper_default_events": len(defaults),
            "technical_errors": sum(r is not None and r["status"] == "error" for _, r in entries),
            "missing": sum(r is None for _, r in entries),
            "tsr_scheduled_available": macro(completed), "tsr_valid_available": macro(valid),
            "tsr_full": macro(completed) if len(defaults) == len(entries) else None,
            "variant_tsr": {v: macro([(j, r) for j, r in valid if j["variant"] == v])
                            for v in sorted({j["variant"] for j, _ in entries})},
            "style_tsr": {s: macro([(j, r) for j, r in valid if r["event"].get("style") == s])
                          for s in ("handin", "slide")} if condition in ("onset_arrival", "mid_arrival", "mid_shift", "remove_return") else {},
            "per_task": [{"task": task, "planned": sum(j["task"] == task for j, _ in entries),
                          "completed": sum(j["task"] == task for j, _ in completed),
                          "valid": sum(j["task"] == task for j, _ in valid),
                          "tsr_valid_available": macro([(j, r) for j, r in valid if j["task"] == task])}
                         for task in sorted({j["task"] for j, _ in entries})]})
    return report


def paired_success_times(manifest):
    manifest = Path(manifest)
    data = json.loads(manifest.read_text())
    rows = {}
    for job in data["jobs"]:
        path = manifest.parent / "episodes" / job["id"] / "result.json"
        if path.exists():
            r = json.loads(path.read_text())
            if r.get("protocol_hash") != job["protocol_hash"]:
                raise ValueError("Protocol mismatch")
            rows[job["method"], job["task"], job["seed"], job["condition"]] = r
    deltas = defaultdict(list)
    for (method, task, seed, condition), row in rows.items():
        clean = rows.get((method, task, seed, "clean"))
        if condition == "clean" or not clean:
            continue
        if not all(r.get("status") == "completed" and r.get("success") and r.get("event", {}).get("valid")
                   and r.get("completion_seconds") is not None for r in (clean, row)):
            continue
        if clean["event"]["initial_scene_sha256"] != row["event"]["initial_scene_sha256"]:
            continue
        deltas[method, condition].append((task, row["completion_seconds"] - clean["completion_seconds"]))
    return [{"method": m, "condition": c, "paired_successes": len(values),
             "mean_added_seconds": mean(delta for _, delta in values),
             "task_macro_added_seconds": mean(mean(d for t, d in values if t == task)
                                               for task in sorted({t for t, _ in values}))}
            for (m, c), values in sorted(deltas.items())]
