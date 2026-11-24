"""Package recorded clean-success / perturbation-failure pairs with their source records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robo_ri.paths import SUBMISSION
from robo_ri.runner import atomic_json


CASES = [
    ("onset_arrival", "task_relevant/suites_seqcount", "t09_s52_mme_vla_onset_sync", "onset_sync"),
    ("mid_arrival", "task_relevant/suites", "t21_s51_mme_vla_arrive_early", "arrive_early"),
    ("mid_shift", "task_relevant/suites", "t21_s51_mme_vla_displace", "displace"),
    ("remove_return", "task_relevant/suites_seqcount", "t09_s52_mme_vla_takeback_early", "takeback_early"),
    ("tv_on", "incidental/suites", "t09_s52_mme_vla_inc_tv", "inc_tv"),
    ("object_passby", "incidental/suites", "t09_s52_mme_vla_inc_passerby", "inc_passerby"),
    ("unrelated_object", "incidental/suites", "t04_s51_mme_vla_inc_place", "inc_place"),
    ("target_like_object", "incidental/suites", "t04_s52_mme_vla_inc_lookalike", "inc_lookalike"),
    ("camera_shake", "incidental/suites", "t09_s52_mme_vla_inc_camshake", "inc_camshake"),
    ("light_change", "incidental/suites", "t09_s52_mme_vla_inc_light", "inc_light"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_at(video, index):
    import cv2
    from PIL import Image
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, min(max(index, 0), count - 1))
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot decode frame {index}: {video}")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), count


def build(out, results):
    from PIL import Image, ImageDraw
    rows = []
    for number, (condition, suite, episode_id, original) in enumerate(CASES, 1):
        episode = results / suite / "mme_vla/episodes" / episode_id
        result = json.loads((episode / "result.json").read_text())
        assert result["status"] == "completed" and result["tsr"] == 0
        clean_id = episode_id.removesuffix(original) + "clean"
        clean = None
        for location in ("task_relevant/suites", "task_relevant/suites_seqcount", "incidental/suites"):
            path = results / location / "mme_vla/episodes" / clean_id
            if (path / "result.json").is_file() and (path / "video.mp4").is_file():
                candidate = json.loads((path / "result.json").read_text())
                if candidate["status"] == "completed" and candidate["tsr"] == 1 and candidate["init_state_sha1"] == result["init_state_sha1"]:
                    clean = path
                    break
        if clean is None:
            raise RuntimeError(f"No matching clean success video: {episode_id}")
        scenario = json.loads((episode / "scenario.json").read_text())
        tr = scenario.get("tr")
        planned = scenario["plan"].get("events", [])
        if tr:
            assert tr["status"] == "done" and not tr.get("skipped") and not tr.get("relocated")
            start, end = tr["start"], tr["end"]
            if condition == "remove_return":
                start = tr["away"]["start"]
            params = tr
        else:
            first = planned[0]
            at = first.get("at", 0)
            start = int(str(at).split("+")[-1]) if at is not None else 0
            end = start + int(first.get("steps", 40))
            params = {"plan_events": planned, "finished_events": scenario.get("events", [])}
        destination = out / f"{number:02d}_{condition}"
        destination.mkdir(parents=True, exist_ok=True)
        for src, target in ((episode / "video.mp4", "failure.mp4"), (clean / "video.mp4", "clean.mp4"),
                            (episode / "result.json", "failure_result.json"), (clean / "result.json", "clean_result.json"),
                            (episode / "scenario.json", "failure_scenario.json")):
            shutil.copy2(src, destination / target)
        pre_frames = result.get("realism", {}).get("prephase_frames", 0)
        instruction = scenario.get("instruction_step") or 0
        def video_index(step):
            return max(0, step - 1 if step < instruction else pre_frames + step - instruction - 10)
        before, _ = frame_at(destination / "failure.mp4", video_index(max(0, start - 10)))
        during, _ = frame_at(destination / "failure.mp4", video_index((start + end) // 2))
        failed, failed_count = frame_at(destination / "failure.mp4", 10**9)
        succeeded, clean_count = frame_at(destination / "clean.mp4", 10**9)
        width, height = before.size
        sheet = Image.new("RGB", (width * 2, (height + 28) * 2), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (image, label) in enumerate(((before, "Before event"), (during, "During event"),
                                               (succeeded, "Clean: success"), (failed, "Perturbed: failure"))):
            x, y = (index % 2) * width, (index // 2) * (height + 28)
            sheet.paste(image, (x, y + 28))
            draw.text((x + 8, y + 8), label, fill="black")
        sheet.save(destination / "comparison.jpg", quality=92)
        missing_stages = [k for k, v in result["stage_done"].items() if not v]
        row = {"number": number, "condition": condition, "method": "FS+Mod (MME-VLA)",
            "task": result["task"], "seed": result["seed"], "clean_success": True, "perturbed_success": False,
            "initial_state_sha1": result["init_state_sha1"], "original_condition": original,
            "source_episode": str(episode.relative_to(SUBMISSION.parent)),
            "source_clean": str(clean.relative_to(SUBMISSION.parent)),
            "source_protocol": scenario.get("version"), "paper_parameter_match": False,
            "usage_note": "Recorded legacy Arena event: same perturbation family; parameters differ from paper Table 2. Qualitative example only.",
            "prompt": result["prompt"], "incomplete_stages": missing_stages,
            "event_steps_in_source_clock": [start, end], "event_parameters": params,
            "video_event_frame_range": [video_index(start), video_index(end)],
            "failure_video": str((destination / "failure.mp4").relative_to(out)),
            "clean_video": str((destination / "clean.mp4").relative_to(out)),
            "comparison": str((destination / "comparison.jpg").relative_to(out)),
            "failure_sha256": sha(destination / "failure.mp4"), "clean_sha256": sha(destination / "clean.mp4"),
            "video_frames": {"failure": failed_count, "clean": clean_count}}
        atomic_json(destination / "case.json", row)
        rows.append(row)
    atomic_json(out / "manifest.json", {"schema": 1, "selection": "qualitative clean-success to failure pairs; not an unbiased TSR sample", "cases": rows})
    with (out / "cases.csv").open("w", newline="") as handle:
        fields = ["number", "condition", "method", "task", "seed", "clean_success", "perturbed_success",
                  "original_condition", "paper_parameter_match", "failure_video", "clean_video"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    captions = ["% Historical qualitative Arena examples; parameters differ from paper defaults."]
    for row in rows:
        label = row["condition"].replace("_", " ")
        captions.append(f"% Video {row['number']:02d}: {row['failure_video']}")
        captions.append(f"\\paragraph{{Video {row['number']:02d}: {label}.}} "
            f"FS+Mod on RoboMemArena task {row['task']}, seed {row['seed']}. "
            "The clean run succeeds and the perturbed run fails under the native task criterion. "
            "Both runs share the same initial scene. This is a historical qualitative example; "
            "its event parameters differ from the default evaluation configuration.\n")
    (out / "captions.tex").write_text("\n".join(captions) + "\n")
    body = ['<!doctype html><meta charset="utf-8"><title>RoboMemArena: ten perturbation failure examples</title>',
            '<style>body{max-width:1100px;margin:40px auto;font:16px system-ui}video{width:49%}img{max-width:100%}section{margin:42px 0}code{word-break:break-all}</style>',
            '<h1>RoboMemArena: ten recorded perturbation failure examples</h1>',
            '<p>Every pair has the same method, task, seed and initial scene hash. Clean succeeds; perturbed execution fails. '
            'These are legacy event recordings. Timing, geometry and transient duration differ from the paper defaults; '
            'they illustrate the event families and are not results of the new paper-configured evaluation.</p>']
    for row in rows:
        body += [f'<section><h2>{row["number"]:02d}. {html.escape(row["condition"])} — task {row["task"]}, seed {row["seed"]}</h2>',
                 '<p>Left: clean success. Right: perturbed failure.</p>',
                 f'<video controls preload="none" src="{row["clean_video"]}"></video> <video controls preload="none" src="{row["failure_video"]}"></video>',
                 f'<p>Incomplete native stages: {html.escape(", ".join(row["incomplete_stages"]))}</p>',
                 f'<img loading="lazy" src="{row["comparison"]}"></section>']
    (out / "index.html").write_text("\n".join(body))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=SUBMISSION / "appendix_videos")
    parser.add_argument("--results", type=Path, default=SUBMISSION.parent / "results")
    arguments = parser.parse_args()
    rows = build(arguments.out.resolve(), arguments.results.resolve())
    print(f"Packaged {len(rows)} verified clean-success / perturbed-failure video pairs")
