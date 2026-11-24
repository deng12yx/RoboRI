"""Deterministic per-control-step events; simulator operations live in scene.py.

No image postprocessing, forced waiting, budget compensation or oracle action is
used. A pending/invalid event is recorded rather than counted as a robustness test.
"""

import numpy as np
from .protocol import CONDITIONS


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


class Perturbation:
    def __init__(self, spec, official_seed, scene):
        self.spec, self.scene = spec, scene
        self.cfg = CONDITIONS.get(spec.condition)
        self.rng = np.random.default_rng(spec.event_seed(official_seed))
        self.log = []
        self.state = "clean" if self.cfg is None else "pending"
        self.object = self.original = self.endpoint = None
        self.required = None
        self.start = self.end = None
        self.saved_visual = None

    def record(self, event, **data):
        self.log.append(dict(event=event, **data))

    def prepare(self):
        if self.cfg is None:
            return
        c = self.spec.condition
        if self.cfg.group == "remember":
            self.object = self.scene.select_task_object(c, self.rng)
            if self.object is None:
                self.state = "inapplicable"
                self.record(
                    "inapplicable", reason="No task-preserving object/stage window"
                )
                return
            self.original = self.scene.position(self.object).copy()
            self.required = self.scene.required_stage(self.object)
            self.endpoint = self.original.copy()
            if c == "shift":
                self.endpoint = self.scene.shift_endpoint(
                    self.object, self.cfg.parameters["distance_m"]
                )
                if self.endpoint is None:
                    self.state = "inapplicable"
                    self.record(
                        "inapplicable",
                        reason="No visible, reachable, collision-free 2 cm endpoint",
                    )
                    return
            if c in ("onset_arrival", "mid_arrival"):
                self.scene.hold(self.object)
                self.scene.move(
                    self.object, self.original + [0, self.cfg.parameters["offset_m"], 0]
                )
                self.state = "absent"
            self.record(
                "selected",
                object=self.object,
                original=self.original.tolist(),
                endpoint=self.endpoint.tolist(),
                role=self.scene.role(self.object),
                required_stage=self.scene.required_stage(self.object),
            )
        else:
            self.scene.prepare_visual(c, self.spec.resolved_subtype, self.rng)

    def before_step(self, native_step, online_step):
        if self.cfg is None or self.state in ("complete", "inapplicable", "invalid"):
            return
        t = native_step if self.cfg.clock == "native" else online_step
        if t < 0:
            return
        c = self.spec.condition
        if self.cfg.group == "ignore":
            if c in ("unrelated", "target_like"):
                change = 48 if self.spec.resolved_subtype == "remove" else 16
                if t == change:
                    self.scene.visual_step(c, t, self.spec.resolved_subtype)
                    self.state = "complete"
                    self.record("complete", step=t, subtype=self.spec.resolved_subtype)
            elif self.cfg.start <= t < self.cfg.end:
                self.scene.visual_step(c, t, self.spec.resolved_subtype)
                if self.state == "pending":
                    self.record("start", step=t)
                self.state = "active"
            elif t >= self.cfg.end and self.state == "active":
                self.scene.restore_visual(c)
                self.state = "complete"
                self.record("complete", step=t)
            return
        if self.start is None:
            due = t >= self.cfg.start
            if c == "onset_arrival":
                due = True
            if not due:
                return
            # A later object is chosen at reset. Never change the selected object
            # just because a different policy traverses the stages differently.
            if not self.scene.can_start(self.object, c):
                if self.scene.use_imminent(self.object):
                    self._invalidate(t, "No admissible pre-use perturbation window")
                return
            self.start = t
            self.end = 20 if c == "onset_arrival" else t + (1 if c == "shift" else 12)
            self.scene.hold(self.object)
            self.state = "active"
            self.record("start", step=t)
        # Deadline advances if the robot approaches earlier than nominal.
        if self.scene.use_imminent(self.object):
            self.end = min(self.end, t)
        if c in ("onset_arrival", "mid_arrival"):
            frac = smooth((t - self.start + 1) / max(1, self.end - self.start))
            p = self.original + [
                0,
                self.cfg.parameters["offset_m"] * (1 - frac),
                self.cfg.parameters["height_m"] * 4 * frac * (1 - frac),
            ]
        elif c == "remove_return":
            p = self.original + (
                np.array(self.cfg.parameters["offset_m"]) if t < self.end else 0
            )
        else:
            p = self.endpoint
        self.scene.move(self.object, p)
        if t >= self.end or c == "shift":
            self.scene.move(self.object, self.endpoint)
            self.scene.release(self.object)
            if not self.scene.endpoint_valid(self.object):
                self._invalidate(t, "Endpoint is not accessible and visible")
                return
            self.state = "complete"
            self.record(
                "complete",
                step=t,
                endpoint=self.endpoint.tolist(),
                displacement_m=float(np.linalg.norm(self.endpoint - self.original)),
            )

    def _invalidate(self, step, reason):
        if self.object is not None and self.original is not None:
            self.scene.move(self.object, self.original)
            self.scene.release(self.object)
        self.state = "invalid"
        self.record("invalid", step=step, reason=reason)

    def after_step(self, online_step):
        if self.spec.condition not in (
            "mid_arrival",
            "shift",
            "remove_return",
        ) or self.state in ("complete", "inapplicable", "invalid"):
            return
        if self.required is not None and self.scene.stage() >= self.required:
            self._invalidate(
                online_step,
                "Required manipulation stage reached before event completion",
            )

    def result(self):
        return dict(
            event_state=self.state,
            event_completed=self.state in ("complete", "clean"),
            event_log=self.log,
        )

    def close(self):
        if self.object is not None:
            if self.state not in ("complete", "clean") and self.original is not None:
                self.scene.move(self.object, self.original)
            self.scene.release(self.object)
        if self.cfg is not None and self.cfg.group == "ignore":
            self.scene.restore_visual(self.spec.condition)
