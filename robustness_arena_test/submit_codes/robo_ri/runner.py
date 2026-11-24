from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np

from .arena import activate
from .baselines import generic_adapter, native_dual
from .perturbations import Perturbation
from .protocol import METHODS
from .paths import SUBMISSION


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def closure_names(fn, seen=None):
    seen = set() if seen is None else seen
    if id(fn) in seen:
        return set()
    seen.add(id(fn))
    out = set()
    for cell in getattr(fn, "__closure__", ()) or ():
        value = cell.cell_contents
        values = list(value.values()) if isinstance(value, dict) else list(value) if isinstance(value, (list, tuple)) else [value]
        for nested in values:
            if isinstance(nested, str):
                out.add(nested.removesuffix("_main"))
            elif callable(nested):
                out.update(closure_names(nested, seen))
    return out


def guarded_specs(specs, event):
    out = []
    for spec in specs:
        names = closure_names(spec.check_fn)
        def check(env, state, start, fn=spec.check_fn, names=names):
            if event.active_pose is not None and event.target in names:
                return False
            return fn(env, state, start)
        out.append(type(spec)(spec.name, check))
    return out


class EpisodeLog(logging.Handler):
    def __init__(self, event):
        super().__init__()
        self.event, self.stages, self.errors, self.exceptions = event, [], [], []

    def emit(self, record):
        message = record.getMessage()
        match = re.search(r"\[t=(\d+)\] (?:Stage completed:|stage done:) (\S+)", message)
        if match:
            self.event.notify_stage(match[2])
            self.stages.append({"recorded_step": self.event.s, "stage": match[2]})
        elif record.levelno >= logging.ERROR:
            self.errors.append(message)
            if record.exc_info:
                self.exceptions.append(record.exc_info[1])


def completion_step(task, stage_names, events, criterion):
    progress = dict.fromkeys(stage_names, False)
    for event in events:
        progress[event["stage"]] = True
        if criterion(task, progress):
            return event["recorded_step"]
    return None


def run_episode(job, config, paths, output, adapter=None):
    from .ri_mem.query import InvalidPlan
    output = Path(output)
    owns_adapter = adapter is None
    output.mkdir(parents=True, exist_ok=True)
    ec, generic = activate(paths["arena_repo"], paths["cache"])
    task, seed, method = job["task"], job["seed"], job["method"]
    np.random.seed(seed)
    env = ec._get_env_class()(bddl_file_name=str(ec._resolve_bddl_path(task)), camera_heights=256,
        camera_widths=256, ignore_done=True, reward_shaping=True, control_freq=20, initialization_noise=None)
    env.seed(seed)
    event = Perturbation(job["condition"], task, seed)
    event.install(env)
    log = EpisodeLog(event)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(log)
    runtime = {**config["runtime"], "seed": seed}
    _, key = ec._resolve_task_id(task)
    prompt = ec.get_prompt(key, ec._resolve_bddl_path(task).stem)
    specs = guarded_specs(generic._task_specs(task), event)
    base_initial = generic._build_initial_state
    import task2_26_reference_stage as stage_eval
    native_initial = stage_eval._build_initial_state
    required_success = stage_eval._stage_success_from_stage_done

    def initial_state(environment):
        state = base_initial(environment)
        for obj in event.objects.values():
            if event.active_pose is None or obj.name != event.target:
                continue
            root = event.env.objects_dict[obj.name].root_body
            if root in state["initial_body_pos"]:
                state["initial_body_pos"][root] = obj.pose[:3].copy()
        return state

    generic._build_initial_state = stage_eval._build_initial_state = initial_state
    started = time.monotonic()
    result = {**job, "schema": 1, "prompt": prompt, "status": "error", "success": False,
              "provenance": config.get("provenance", {}), "max_steps": runtime["max_steps"]}
    try:
        if method in ("predimem", "traceflow"):
            score, stages, success, diagnostics, front, wrist = native_dual(method, paths,
                config["backends"][method], task, env, prompt, output, runtime, specs, logging.getLogger())
        else:
            adapter = adapter or generic_adapter(method, paths, config["backends"].get(method, {}))
            if hasattr(adapter, "seed_episode"):
                adapter.seed_episode(seed)
            adapter.reset()
            score, stages, success, diagnostics, front, wrist = generic.run_episode_with_stateful_stages(
                task_id=task, env=env, adapter=adapter, prompt=prompt, resize_size=256,
                replan_steps=METHODS[method]["chunk"], num_steps_wait=10, max_steps=runtime["max_steps"],
                post_goal_steps=runtime["post_goal_steps"], stage_specs=specs,
                goal_monitor_dict=ec._build_goal_monitor_dict(ec._resolve_bddl_path(task)),
                goal_check_override=generic._goal_override_check(task), fail_on_extra_pour=True,
                extra_pour_monitor_steps=30)
        if log.errors:
            if any(isinstance(e, InvalidPlan) for e in log.exceptions):
                raise next(e for e in log.exceptions if isinstance(e, InvalidPlan))
            raise RuntimeError("; ".join(log.errors))
        result.update(status="completed", success=bool(success), csr=float(score), diagnostics=diagnostics,
                      stage_done=stages, stage_events=log.stages)
        if config.get("video", False) and front:
            import imageio.v2 as imageio
            with imageio.get_writer(output / "video.mp4", fps=20) as writer:
                for main, hand in zip(front, wrist):
                    writer.append_data(np.concatenate([main, hand], axis=1))
        result["online_steps"] = len(front)
        successful_step = completion_step(task, stages.keys(), log.stages, required_success) if success else None
        result["completion_recorded_step"] = successful_step
        result["completion_seconds"] = (successful_step - 10) / 20 if successful_step is not None else None
    except InvalidPlan as error:
        result.update(status="completed", success=False, failure_reason="invalid_query_plan", error=str(error))
    except Exception as error:
        import traceback
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["event"] = event.report() if hasattr(event, "scene_hash") else {"valid": False, "invalid_reason": "initialization error"}
        generic._build_initial_state, stage_eval._build_initial_state = base_initial, native_initial
        logging.getLogger().removeHandler(log)
        if owns_adapter and adapter is not None and hasattr(adapter, "close"):
            adapter.close()
        event.close()
        env.close()
        atomic_json(output / "result.json", result)
    return result


def run_manifest(manifest, paths, method=None, limit=None):
    from contextlib import ExitStack
    with ExitStack() as cleanup:
        return _run_jobs(manifest, paths, method, limit, cleanup)


def _run_jobs(manifest, paths, method, limit, cleanup):
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text())
    for name, expected in data["config"].get("provenance", {}).get("source_sha256", {}).items():
        if hashlib.sha256((SUBMISSION / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed since planning: {name}; create a new run manifest")
    count = 0
    adapters = {}
    for job in data["jobs"]:
        if method and job["method"] != method:
            continue
        output = manifest.parent / "episodes" / job["id"]
        result_file = output / "result.json"
        if result_file.exists():
            old = json.loads(result_file.read_text())
            if old.get("protocol_hash") != job["protocol_hash"]:
                raise RuntimeError("Existing episode belongs to a different protocol")
            if old["status"] == "completed":
                continue
        try:
            adapter = None
            if job["method"] not in ("predimem", "traceflow"):
                if job["method"] not in adapters:
                    activate(paths["arena_repo"], paths["cache"])
                    adapter = generic_adapter(job["method"], paths, data["config"]["backends"].get(job["method"], {}))
                    if hasattr(adapter, "close"):
                        cleanup.callback(adapter.close)
                    adapters[job["method"]] = adapter
                adapter = adapters[job["method"]]
            result = run_episode(job, data["config"], paths, output, adapter=adapter)
        except Exception as error:
            import traceback
            result = {**job, "schema": 1, "status": "error", "success": False,
                      "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                      "event": {"valid": False, "invalid_reason": "episode initialization failed"}}
            atomic_json(output / "result.json", result)
        print(json.dumps({k: result.get(k) for k in ("id", "status", "success", "error")}), flush=True)
        count += 1
        if result["status"] == "error":
            raise RuntimeError(f"Episode failed technically: {job['id']}; see {output / 'result.json'}")
        if limit and count >= limit:
            break
