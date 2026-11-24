"""Check event mechanics in MuJoCo. Scripted triggers here are not evaluation episodes."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robo_ri.paths import load_paths
from robo_ri.arena import activate
from robo_ri.perturbations import Perturbation
from robo_ri.protocol import CONDITIONS
from robo_ri.runner import atomic_json

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--gpu", default="4")
parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
parser.add_argument("--seed", type=int, default=50)
args = parser.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, MUJOCO_EGL_DEVICE_ID=args.gpu,
                  OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
paths = load_paths()
ec, generic = activate(paths["arena_repo"], paths["cache"])
import mujoco
import numpy as np

results = []
for condition in args.conditions:
    np.random.seed(args.seed)
    env = ec._get_env_class()(bddl_file_name=str(ec._resolve_bddl_path(25)), camera_heights=256,
        camera_widths=256, ignore_done=True, reward_shaping=True, control_freq=20, initialization_noise=None)
    env.seed(args.seed)
    event = Perturbation(condition, 25, args.seed)
    event.install(env)
    try:
        obs = env.reset()
        release_pose = None
        for s in range(290):
            if s == 55:
                first = event.objects["butter_1"]
                event.d.qpos[first.qadr + 2] += .05
                mujoco.mj_forward(event.m, event.d)
            if s == 60:
                event.notify_stage("mechanics_test_first_stage")
            obs, _, _, _ = env.step(ec.LIBERO_DUMMY_ACTION)
            if event.ended is not None and release_pose is None:
                release_pose = event._pos(event.target).copy()
        report = event.report()
        assert report["valid"], report
        if condition in ("onset_arrival", "mid_arrival", "remove_return"):
            reference = event.saved[:3] if condition == "remove_return" else event.objects[event.target].pose[:3]
            np.testing.assert_allclose(release_pose, reference, atol=1e-9)
            np.testing.assert_allclose(event._pos(event.target)[:2], reference[:2], atol=.005)
        if condition == "mid_shift":
            distance = np.linalg.norm(event._pos(event.target)[:2] - event.objects[event.target].pose[:2])
            expected = .025 if ((args.seed - 50) // 2) % 2 == 0 else .04
            assert abs(distance - expected) < .003, distance
        if condition in ("camera_shake", "light_change"):
            assert (report["end"] - report["start"]) / 20 < .2
            np.testing.assert_array_equal(event.m.cam_pos, event.camera_pos)
            np.testing.assert_array_equal(event.m.light_ambient, event.ambient)
            np.testing.assert_array_equal(event.m.light_diffuse, event.diffuse)
        from PIL import Image
        image = np.asarray(obs["agentview_image"])[::-1]
        (args.out / condition).mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(args.out / condition / "last.png")
        atomic_json(args.out / condition / "mechanics.json", report)
        results.append({"condition": condition, "passed": True, "event": report})
        print(condition, "passed", flush=True)
    except Exception as error:
        import traceback
        results.append({"condition": condition, "passed": False, "error": repr(error), "traceback": traceback.format_exc()})
        print(condition, repr(error), flush=True)
    finally:
        event.close()
        env.close()
    atomic_json(args.out / "summary.json", {"type": "mechanics_test_only", "results": results})
if not all(r["passed"] for r in results):
    raise SystemExit(1)
