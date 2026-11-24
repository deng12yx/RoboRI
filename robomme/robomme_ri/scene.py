"""ManiSkill scene port. Uses task metadata only for event eligibility/auditing.

Policy and RI-Mem receive RGB/proprioception/instruction only. Visual distractors
have no collision geometry. Existing object physics are restored on completion.
"""

import re
import numpy as np


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pose(p, q=(1, 0, 0, 0)):
    import sapien

    return sapien.Pose(
        np.asarray(p).reshape(3).tolist(), np.asarray(q).reshape(4).tolist()
    )


class Scene:
    def __init__(self, env):
        self.u = env.unwrapped
        self.objects = dict(self.u.scene.actors)
        self.objects.update(
            {
                n: a
                for n, a in self.u.scene.articulations.items()
                if a is not self.u.agent.robot
            }
        )
        self.objects = {
            n: a
            for n, a in self.objects.items()
            if n not in ("ground", "table-workspace")
            and not n.startswith("ri_")
            and "highlight" not in n
        }
        self.saved = {}
        self.visuals = []
        self.orig_camera = None
        self.orig_ambient = None
        self.hidden_pairs = {}
        for n in self.objects:
            if n.startswith("bin_"):
                p = self.position(n)
                self.hidden_pairs[n] = [
                    (m, self.position(m) - p)
                    for m in self.objects
                    if "cube" in m
                    and np.linalg.norm(self.position(m)[:2] - p[:2]) < 0.03
                    and self.position(m)[2] < p[2]
                ]

    def position(self, name):
        return array(self.objects[name].pose.p).reshape(-1, 3)[0]

    def quaternion(self, name):
        return array(self.objects[name].pose.q).reshape(-1, 4)[0]

    def _body_list(self, name):
        return list(getattr(self.objects[name], "_bodies", []))

    def hold(self, name):
        if name in self.saved:
            return
        self.saved[name] = (
            self.quaternion(name).copy(),
            [(b, bool(b.kinematic)) for b in self._body_list(name)],
        )
        for b, _ in self.saved[name][1]:
            b.set_kinematic(True)
        for child, _ in self.hidden_pairs.get(name, []):
            self.hold(child)

    def move(self, name, p):
        obj = self.objects[name]
        q = self.saved[name][0] if name in self.saved else self.quaternion(name)
        if hasattr(obj, "links"):
            obj.set_root_pose(pose(p, q))
            obj.set_root_linear_velocity(np.zeros(3))
            obj.set_root_angular_velocity(np.zeros(3))
        else:
            obj.set_pose(pose(p, q))
        for child, offset in self.hidden_pairs.get(name, []):
            self.move(child, np.asarray(p) + offset)

    def release(self, name):
        saved = self.saved.pop(name, None)
        if saved:
            for body, was_kinematic in saved[1]:
                body.set_linear_velocity(np.zeros(3))
                body.set_angular_velocity(np.zeros(3))
                body.set_kinematic(was_kinematic)
        for child, _ in self.hidden_pairs.get(name, []):
            self.release(child)

    def segment_name(self, segment):
        if isinstance(segment, (list, tuple)):
            segment = segment[0] if segment else None
        for n, a in self.objects.items():
            if a is segment or getattr(segment, "articulation", None) is a:
                return n
        return None

    def tasks(self):
        return list(getattr(self.u, "task_list", []) or [])

    def task_object(self, task):
        name = self.segment_name(task.get("segment"))
        if name is not None:
            return name
        # PatternLock/RouteStick omit segment but bind the target in lambdas.
        for field in ("solve", "func"):
            functions = task.get(field)
            functions = (
                functions if isinstance(functions, (list, tuple)) else [functions]
            )
            for fn in functions:
                for value in getattr(fn, "__defaults__", None) or ():
                    name = self.segment_name(value)
                    if name is not None:
                        return name
        return None

    def stage(self):
        return int(getattr(self.u, "current_task_index", 0) or 0)

    def required_stage(self, name):
        return next(
            (
                i
                for i, t in enumerate(self.tasks())
                if i >= self.stage() and self.task_object(t) == name
            ),
            None,
        )

    def role(self, name):
        return (
            "target"
            if any(self.task_object(t) == name for t in self.tasks())
            else "nontarget"
        )

    def tcp_distance(self, name):
        return float(
            np.linalg.norm(
                array(self.u.agent.tcp.pose.p).reshape(-1, 3)[0] - self.position(name)
            )
        )

    def use_imminent(self, name):
        stage = self.required_stage(name)
        return (
            stage is not None
            and stage <= self.stage()
            and self.tcp_distance(name) < 0.18
        )

    def can_start(self, name, condition):
        stage = self.required_stage(name)
        if self.tcp_distance(name) < 0.18:
            return False
        if condition == "onset_arrival":
            return True
        # stage 0 may be a watch, a button, or a first manipulation. An event
        # must occur after the first actual non-demo manipulation has begun.
        active = self.tasks()[self.stage()] if self.stage() < len(self.tasks()) else {}
        first = next(
            (
                i
                for i, t in enumerate(self.tasks())
                if not t.get("demonstration") and self.task_object(t) is not None
            ),
            None,
        )
        return (
            first is not None
            and self.stage() >= first
            and not active.get("demonstration", False)
            and (stage is None or stage > self.stage())
        )

    def select_task_object(self, condition, rng):
        names = []
        for t in self.tasks():
            n = self.task_object(t)
            if n and n not in names:
                names.append(n)
        if not names:
            return None
        # Prevent revealing hidden answers and moving task clocks/dense point
        # patterns. Such cases require a task-specific admissibility extension.
        task = self.u.spec.id
        if task == "StopCube":
            return None
        candidates = [
            n
            for n in names
            if self.position(n)[2] < 0.3 and self.tcp_distance(n) >= 0.18
        ]
        if condition == "onset_arrival":
            return candidates[0] if candidates else None
        if condition in ("shift", "remove_return"):

            def kind(name):
                for pattern, category in (
                    (r"cube|fixed_cube|target_cube", "cube"),
                    (r"bin_", "container"),
                    (r"button", "button"),
                    (r"target_\d|^target$", "target"),
                    (r"peg", "peg"),
                    (r"board_with_hole|box_with_hole", "receptacle"),
                ):
                    if re.search(pattern, name):
                        return category
                return name

            kinds = {kind(n) for n in names}
            covered = {
                child for pairs in self.hidden_pairs.values() for child, _ in pairs
            }
            candidates = [
                n
                for n in self.objects
                if kind(n) in kinds
                and n not in covered
                and self.position(n)[2] < 0.3
                and self.tcp_distance(n) >= 0.18
                and self.visible(self.position(n))
            ]
        candidates = [
            n
            for n in candidates
            if self.required_stage(n) is None or self.required_stage(n) > self.stage()
        ]
        if condition == "mid_arrival":
            candidates = [n for n in candidates if self.required_stage(n) is not None]
        if condition in ("mid_arrival", "remove_return"):
            candidates = [
                n
                for n in candidates
                if not n.startswith("bin_") and "target_cube" not in n
            ]
        # Choice is fixed at reset, independent of policy RNG and behavior.
        return candidates[int(rng.integers(len(candidates)))] if candidates else None

    def visible(self, p):
        camera = self.u._sensors["base_camera"]
        cp = camera.camera.get_model_matrix()
        mat = array(cp).reshape(-1, 4, 4)[0]
        local = np.linalg.inv(mat) @ np.r_[p, 1]
        # get_model_matrix uses OpenGL camera coordinates (-z forward).
        depth = -local[2]
        return bool(
            depth > 0.01
            and abs(local[0] / depth) < 0.90
            and abs(local[1] / depth) < 0.90
        )

    def endpoint_valid(self, name):
        p = self.position(name)
        base = array(self.u.agent.robot.pose.p).reshape(-1, 3)[0]
        return (
            self.visible(p)
            and np.linalg.norm(p[:2] - base[:2]) < 0.8
            and -0.45 < p[0] < 0.30
            and abs(p[1]) < 0.45
        )

    def shift_endpoint(self, name, distance):
        p = self.position(name)
        others = [
            self.position(n)
            for n in self.objects
            if n != name
            and n not in {m for m, _ in self.hidden_pairs.get(name, [])}
            and self.position(n)[2] < 0.3
        ]
        choices = []
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = p + [d[0] * distance, d[1] * distance, 0]
            clearance = min(
                (np.linalg.norm(q[:2] - o[:2]) for o in others), default=1.0
            )
            old_clearance = min(
                (np.linalg.norm(p[:2] - o[:2]) for o in others), default=1.0
            )
            minimum = min(0.08, old_clearance)
            if (
                -0.42 < q[0] < 0.26
                and abs(q[1]) < 0.4
                and clearance >= minimum
                and self.visible(q)
            ):
                choices.append((clearance, q))
        return max(choices, key=lambda x: x[0])[1] if choices else None

    def _box(self, name, half, color, p):
        import sapien

        b = self.u.scene.create_actor_builder()
        material = sapien.render.RenderMaterial(base_color=color)
        b.add_box_visual(half_size=list(half), material=material)
        b.set_initial_pose(pose(p))
        actor = b.build_kinematic(name="ri_" + name)
        self.visuals.append(actor)
        return actor

    def prepare_visual(self, condition, subtype, rng):
        if condition == "tv":
            self.tv = [
                self._box(
                    f"tv{i}",
                    (0.004, 0.045, 0.055),
                    (1, 0, 0, 1),
                    (-0.35, (i - 1.5) * 0.10, 0.28),
                )
                for i in range(4)
            ]
            for a in self.tv:
                a.set_pose(pose((-10, -10, -10)))
        elif condition == "passby":
            self.item = self._box(
                "passer", (0.025,) * 3, (0.9, 0.2, 0.7, 1), (-10,) * 3
            )
        elif condition in ("unrelated", "target_like"):
            color = (
                (0.1, 0.8, 0.2, 1) if condition == "target_like" else (0.6, 0.2, 0.8, 1)
            )
            self.item = self._box("object", (0.025,) * 3, color, (-10,) * 3)
            self.cross = []
            self.spot = np.array([0.18, 0.28, 0.025])
            if condition == "target_like":
                self.cross = [
                    self._box(f"cross{i}", half, (1, 1, 1, 1), (-10,) * 3)
                    for i, half in enumerate(
                        ((0.026, 0.004, 0.004), (0.004, 0.026, 0.004))
                    )
                ]
            if subtype == "remove":
                self._place_item(True)
        elif condition == "camera":
            self.camera = self.u._sensors["base_camera"].camera
            self.orig_camera = self.camera.get_local_pose()
        elif condition == "light":
            self.orig_ambient = list(self.u.scene.ambient_light)

    def _place_item(self, show):
        self.item.set_pose(pose(self.spot if show else (-10, -10, -10)))
        for a in self.cross:
            a.set_pose(pose(self.spot + [0, 0, 0.026] if show else (-10, -10, -10)))

    def visual_step(self, condition, t, subtype):
        if condition == "tv":
            palette = (
                (1, 0.1, 0.1, 1),
                (0.1, 1, 0.1, 1),
                (0.1, 0.1, 1, 1),
                (1, 1, 0.1, 1),
            )
            # Move colored blocks between four screen positions every 8 steps.
            for i, a in enumerate(self.tv):
                a.set_pose(
                    pose((-0.35, (((i + (t - 16) // 8) % 4) - 1.5) * 0.10, 0.28))
                )
                for obj in a._objs:
                    for comp in obj.components:
                        if hasattr(comp, "render_shapes"):
                            for shape in comp.render_shapes:
                                for part in shape.parts:
                                    part.material.base_color = palette[
                                        (i + (t - 16) // 8) % 4
                                    ]
        elif condition == "passby":
            self.item.set_pose(pose((-0.30, -0.34 + 0.68 * (t - 32) / 64, 0.22)))
        elif condition in ("unrelated", "target_like"):
            self._place_item(subtype == "insert")
        elif condition == "camera":
            p = array(self.orig_camera.p).reshape(-1, 3)[0].copy()
            p[1] += 0.018 * np.sin(2 * np.pi * (t - 16 + 1) / 3)
            self.camera.set_local_pose(
                pose(p, array(self.orig_camera.q).reshape(-1, 4)[0])
            )
        elif condition == "light":
            self.u.scene.set_ambient_light(
                [2, 2, 2] if subtype == "bright" else [0, 0, 0]
            )

    def restore_visual(self, condition):
        if condition == "camera" and self.orig_camera is not None:
            self.camera.set_local_pose(
                pose(
                    array(self.orig_camera.p).reshape(-1, 3)[0],
                    array(self.orig_camera.q).reshape(-1, 4)[0],
                )
            )
        if condition == "light" and self.orig_ambient is not None:
            self.u.scene.set_ambient_light(self.orig_ambient)
        for a in self.visuals:
            a.set_pose(pose((-10, -10, -10)))
