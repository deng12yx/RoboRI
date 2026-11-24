"""Matched clean/perturbed closed-loop evaluation on official RoboMME seeds."""

from collections import deque
import json
from pathlib import Path
import time
import traceback
import numpy as np
from .perturbations import Perturbation
from .scene import Scene, array
from .policies import instruction
from .rimem.query import InvalidPlan


class EventHook:
    """Hook BaseEnv.step, including native actions inside waypoint planners.

    Native initialization/demo steps and online control steps have distinct clocks.
    The outer evaluator can execute chunks without skipping any event clock ticks.
    """

    def __init__(self, env, spec, seed):
        self.env, self.u, self.spec, self.seed = env, env.unwrapped, spec, seed
        self.native = self.online = 0
        self.initializing = True
        self.event = None
        self.original = self.u.step
        self.had_local = "step" in self.u.__dict__
        self.original_local = self.u.__dict__.get("step")

        def step(action, *args, **kwargs):
            if self.event is None:
                self.event = Perturbation(spec, seed, Scene(env))
                self.event.prepare()
            self.event.before_step(
                self.native, -1 if self.initializing else self.online
            )
            result = self.original(action, *args, **kwargs)
            self.native += 1
            if not self.initializing:
                self.online += 1
                self.event.after_step(self.online)
            return result

        self.u.step = step

    def close(self):
        if self.event:
            self.event.close()
        if self.had_local:
            self.u.step = self.original_local
        else:
            del self.u.step


def append_jsonl(path, row):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def run_episode(spec, policy, output, save_video=False, config_hash=None):
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    out = Path(output)
    start = time.monotonic()
    native_budget = getattr(policy, "max_control_steps", spec.max_steps)
    builder = BenchmarkEnvBuilder(
        env_id=spec.task,
        dataset="test",
        action_space=policy.action_space,
        gui_render=False,
        max_steps=native_budget,
    )
    seed, difficulty = builder.resolve_episode(spec.episode)
    env = builder.make_env_for_episode(
        spec.episode, max_steps=native_budget, **getattr(policy, "env_kwargs", {})
    )
    hook = EventHook(env, spec, int(seed))
    writer = None
    row = {
        **spec.as_dict(),
        "key": spec.key(),
        "official_seed": int(seed),
        "difficulty": str(difficulty),
        "protocol": "robomme-ri-paper-v1",
        "config_hash": config_hash,
        "policy": policy.metadata,
        "evaluation_kind": getattr(policy, "evaluation_kind", "policy"),
    }
    try:
        obs, info = env.reset()
        hook.initializing = False
        row["instruction"] = instruction(info)
        row["initialization_steps"] = hook.native
        policy.reset(spec, obs, info, env)
        if save_video:
            import imageio.v2 as imageio

            video = out / "videos" / (spec.key() + ".mp4")
            video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                str(video), fps=20, codec="libx264", macro_block_size=1
            )
            row["video"] = str(video.resolve())

        def record(o):
            if writer:
                for f in o["front_rgb_list"]:
                    writer.append_data(np.asarray(f, np.uint8))

        record(obs)
        actions = deque()
        decisions = 0
        status = "timeout"
        while hook.online < native_budget and decisions < getattr(
            policy, "max_decisions", 100000
        ):
            if not actions:
                chunk = np.asarray(policy.act(obs, info), np.float32)
                if chunk.ndim != 2 or not len(chunk) or not np.isfinite(chunk).all():
                    raise ValueError("Policy returned an invalid action chunk")
                actions.extend(
                    chunk[
                        : min(
                            spec.exec_horizon,
                            getattr(policy, "exec_horizon", spec.exec_horizon),
                        )
                    ]
                )
                decisions += 1
            action = actions.popleft()
            if hasattr(policy, "convert_action"):
                action = policy.convert_action(action, obs)
            obs, _, term, trunc, info = env.step(action)
            record(obs)
            policy.observe(obs)
            raw_status = info.get("status")
            if isinstance(raw_status, list):
                raw_status = raw_status[-1]
            if raw_status in ("success", "fail", "timeout", "error"):
                status = "technical_error" if raw_status == "error" else raw_status
                break
            if bool(array(term).any()) or bool(array(trunc).any()):
                status = (
                    "success"
                    if bool(getattr(_find_demo(env), "episode_success", False))
                    else "timeout"
                    if bool(array(trunc).any())
                    else "fail"
                )
                break
        row.update(
            status=status,
            decisions=decisions,
            online_steps=hook.online,
            execution_seconds=hook.online / 20.0,
        )
    except InvalidPlan as e:
        row.update(
            status="invalid_query", error=str(e)
        )  # unsuccessful enhanced episode
    except Exception:
        row.update(status="technical_error", error=traceback.format_exc())
    finally:
        if writer:
            writer.close()
        if hook.event:
            row.update(hook.event.result())
        else:
            row.update(
                event_state="not_initialized", event_completed=spec.condition == "clean"
            )
        if getattr(policy, "rimem", None):
            path = out / "traces" / (spec.key() + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(policy.rimem.trace, indent=2) + "\n")
            row["rimem_trace"] = str(path.resolve())
            policy.rimem.close()
        hook.close()
        env.close()
        row["wall_seconds"] = time.monotonic() - start
    append_jsonl(out / "rollouts.jsonl", row)
    return row


def _find_demo(env):
    while env is not None:
        if type(env).__name__ == "DemonstrationWrapper":
            return env
        env = getattr(env, "env", None)


def summarize(rows):
    """Main Eq.7 macro TSR; also expose coverage and technical/invalid counts."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in rows:
        if r.get("evaluation_kind", "policy") != "policy" or "test_controller" in r.get(
            "policy", {}
        ):
            continue
        groups[(r["method"], r["condition"], r.get("config_hash"))].append(r)
    result = []
    for (method, condition, config_hash), group in sorted(
        groups.items(), key=lambda x: str(x[0])
    ):
        # Latest valid attempt per task/episode/config; never inflate N on retries.
        unique = {
            (r["task"], r["episode"], r.get("subtype"), r.get("model_seed")): r
            for r in group
        }
        valid = [
            r
            for r in unique.values()
            if r["status"] != "technical_error"
            and (r.get("event_completed") or r["status"] == "invalid_query")
        ]
        tasks = defaultdict(list)
        for r in valid:
            tasks[r["task"]].append(r["status"] == "success")
        rates = {t: 100.0 * sum(s) / len(s) for t, s in tasks.items()}
        result.append(
            dict(
                method=method,
                condition=condition,
                config_hash=config_hash,
                macro_tsr=np.mean(list(rates.values())).item() if rates else None,
                episodes=len(valid),
                tasks=len(tasks),
                per_task=rates,
                planned=len(unique),
                technical_errors=sum(
                    r["status"] == "technical_error" for r in unique.values()
                ),
                inadmissible_events=sum(
                    not r.get("event_completed") and r["status"] != "invalid_query"
                    for r in unique.values()
                ),
                full_16x50_coverage=len(tasks) == 16
                and all(len(v) == 50 for v in tasks.values()),
            )
        )
    return result
