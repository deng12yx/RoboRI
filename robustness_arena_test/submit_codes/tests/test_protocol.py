import json

import pytest

from robo_ri.protocol import CONDITIONS, jobs
from robo_ri.analysis import summarize, paired_success_times
from robo_ri.runner import atomic_json


def test_full_matrix_includes_clean_separately():
    from robo_ri.paths import SUBMISSION
    cfg = json.loads((SUBMISSION / "configs/paper.json").read_text())
    matrix = jobs(cfg)
    assert len(CONDITIONS) == 10
    assert len(matrix) == 26 * 51 * 11 * 9
    assert len({j["id"] for j in matrix}) == len(matrix)
    assert sum(j["condition"] != "clean" for j in matrix) == 26 * 51 * 10 * 9
    cfg["conditions"] += ["arrive_late"]
    with pytest.raises(ValueError):
        jobs(cfg)


def test_macro_tsr_and_coverage_are_not_pooled(tmp_path):
    rows = []
    for task, outcomes in ((2, [True]), (3, [False, False, True])):
        for seed, success in enumerate(outcomes):
            job = dict(id=f"t{task}s{seed}", method="fs_mod", task=task, seed=seed,
                       condition="mid_shift", variant="default", protocol_hash="protocol")
            rows.append(job)
            atomic_json(tmp_path / "episodes" / job["id"] / "result.json",
                        {**job, "status": "completed", "success": success, "event": {"valid": True, "style": "slide"}})
    atomic_json(tmp_path / "manifest.json", {"jobs": rows})
    report = summarize(tmp_path / "manifest.json")[0]
    assert report["tsr_full"] == pytest.approx((1 + 1 / 3) / 2 * 100)
    assert report["tsr_full"] != 50
    rows.append({**rows[-1], "id": "missing", "seed": 99})
    atomic_json(tmp_path / "manifest.json", {"jobs": rows})
    report = summarize(tmp_path / "manifest.json")[0]
    assert report["tsr_full"] is None and report["missing"] == 1


def test_time_comparison_rejects_scene_mismatch(tmp_path):
    from robo_ri.runner import completion_step
    events = [{"stage": "place", "recorded_step": 200}, {"stage": "optional_close", "recorded_step": 230}]
    assert completion_step(4, ["place", "optional_close"], events, lambda _, done: done["place"]) == 200
    assert completion_step(6, ["place", "optional_close"], events, lambda _, done: all(done.values())) == 230
    rows = []
    for condition, duration, scene in (("clean", 12., "same"), ("mid_shift", 17., "different")):
        job = dict(id=condition, method="fs_mod", task=21, seed=50, condition=condition, protocol_hash="p")
        rows.append(job)
        atomic_json(tmp_path / "episodes" / condition / "result.json", {**job,
            "status": "completed", "success": True, "completion_seconds": duration,
            "event": {"valid": True, "initial_scene_sha256": scene}})
    atomic_json(tmp_path / "manifest.json", {"jobs": rows})
    assert paired_success_times(tmp_path / "manifest.json") == []
    result_file = tmp_path / "episodes/mid_shift/result.json"
    row = json.loads(result_file.read_text())
    row["event"]["initial_scene_sha256"] = "same"
    atomic_json(result_file, row)
    assert paired_success_times(tmp_path / "manifest.json")[0]["mean_added_seconds"] == 5
