from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .protocol import CONDITIONS, variant


# First manipulated object and a later required object or receptacle, in native BDDL names.
TARGETS = {
    1: ("cookies_1", "tomato_sauce_1"), 2: ("butter_1", "popcorn_1"),
    3: ("cream_cheese_1", "chocolate_pudding_1"), 4: ("butter_1", "butter_1"),
    5: ("butter_1", "butter_1"), 6: ("tomato_sauce_1", "cookies_1"),
    7: ("tomato_sauce_1", "frypan_1"), 8: ("chocolate_pudding_1", "tomato_sauce_1"),
    9: ("butter_1", "tomato_sauce_1"), 10: ("wine_bottle_1", "white_yellow_mug_1"),
    11: ("cookies_1", "butter_1"), 12: ("cookies_1", "chocolate_pudding_1"),
    13: ("cookies_1", "butter_1"), 14: ("cookies_1", "chocolate_pudding_1"),
    15: ("butter_1", "milk_1"), 16: ("milk_1", "red_coffee_mug_1"),
    17: ("butter_1", "chocolate_pudding_1"), 18: ("chocolate_pudding_1", "butter_1"),
    19: ("tomato_sauce_1", "milk_1"), 20: ("cookies_1", "chocolate_pudding_1"),
    21: ("butter_1", "chocolate_pudding_1"), 22: ("tomato_sauce_1", "cookies_1"),
    23: ("cream_cheese_1", "popcorn_1"), 24: ("cookies_1", "popcorn_1"),
    25: ("butter_1", "cream_cheese_1"), 26: ("chocolate_pudding_1", "cream_cheese_1"),
}


def _geom(body, name, kind, size, rgba, pos="0 0 0", **kwargs):
    return ET.SubElement(body, "geom", name=name, type=kind, size=size, pos=pos, rgba=rgba,
                         contype="0", conaffinity="0", group="1", **kwargs)


def inject(model):
    world = model.worldbody
    for name in ("ri_object", "ri_hand", "ri_person"):
        body = ET.SubElement(world, "body", name=name, mocap="true", pos="-3 0 2")
        if name == "ri_object":
            _geom(body, "ri_sphere", "sphere", ".034", ".10 .25 .95 1")
            _geom(body, "ri_box", "box", ".029 .020 .025", ".8 .7 .2 0")
            _geom(body, "ri_cross_h", "box", ".018 .001 .003", "1 1 1 0", pos="0 -.021 0")
            _geom(body, "ri_cross_v", "box", ".003 .001 .018", "1 1 1 0", pos="0 -.021 0")
        elif name == "ri_hand":
            _geom(body, "ri_palm", "box", ".045 .04 .013", ".86 .69 .57 1")
            _geom(body, "ri_arm", "capsule", ".03", ".2 .3 .4 1", fromto=".05 0 0 .35 0 .1")
        else:
            _geom(body, "ri_torso", "capsule", ".16", ".55 .2 .18 1", fromto="0 0 .9 0 0 1.35")
            _geom(body, "ri_head", "sphere", ".1", ".86 .69 .57 1", pos="0 0 1.6")
            for sign in (-1, 1):
                _geom(body, f"ri_leg_{sign}", "capsule", ".065", ".16 .18 .24 1",
                      fromto=f"0 {sign * .09} .9 0 {sign * .09} .1")
    for i in range(8):
        _geom(world, f"ri_stripe_{i}", "box", ".003 .052 .22", ".05 .05 .05 0",
              pos=f"-1.727 {0.625 + (i - 3.5) * .105} .577")


@dataclass
class FreeObject:
    name: str
    pose: np.ndarray
    qadr: int
    vadr: int
    geoms: np.ndarray
    collision: np.ndarray


class Perturbation:
    """One environment-level event on the recorded 20 Hz clock, including ten settling steps."""

    def __init__(self, condition, task, seed, style=None):
        if condition != "clean" and condition not in CONDITIONS:
            raise ValueError(condition)
        self.condition, self.task, self.seed = condition, task, seed
        self.style = style or ("handin" if seed % 2 == 0 else "slide")
        self.variant = variant(condition, seed)
        self.rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{task}/{seed}/{condition}".encode()).digest()[:8], "little"))
        self.s = self.stage_count = 0
        self.started = self.ended = self.first_manipulation = self.target_manipulation = None
        self.invalid = None
        self.active_pose = None
        self.events = []
        self.visible_steps = 0
        self.objects = {}
        self.target = TARGETS[task][0 if condition == "onset_arrival" else 1]

    def install(self, env):
        self.env = env.env if hasattr(env, "env") else env
        inner = self.env
        load, reset, pre, post = inner._load_model, inner._reset_internal, inner._pre_action, inner._post_action
        self.originals = load, reset, pre, post

        def load_model():
            load()
            inject(inner.model)

        def reset_internal():
            reset()
            self._reset()

        def pre_action(action, policy_step=False):
            pre(action, policy_step)
            self._apply()

        def post_action(action):
            result = post(action)
            self.s += 1
            self._tick()
            return result

        inner._load_model, inner._reset_internal = load_model, reset_internal
        inner._pre_action, inner._post_action = pre_action, post_action

    def close(self):
        if hasattr(self, "originals"):
            self.env._load_model, self.env._reset_internal, self.env._pre_action, self.env._post_action = self.originals

    def _reset(self):
        import mujoco
        self.m, self.d = self.env.sim.model._model, self.env.sim.data._data
        self.scene_hash = hashlib.sha256(self.d.qpos.round(6).tobytes()).hexdigest()
        self.initial_qpos = self.d.qpos.copy()
        self.camera_pos, self.camera_quat = self.m.cam_pos.copy(), self.m.cam_quat.copy()
        self.ambient, self.diffuse = self.m.light_ambient.copy(), self.m.light_diffuse.copy()
        for name, obj in self.env.objects_dict.items():
            if not obj.joints:
                continue
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, obj.joints[-1])
            if jid < 0 or self.m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            root = self.env.sim.model.body_name2id(obj.root_body)
            descendants = {root}
            for bid in range(root + 1, self.m.nbody):
                if self.m.body_parentid[bid] in descendants:
                    descendants.add(bid)
            geoms = np.flatnonzero(np.isin(self.m.geom_bodyid, list(descendants)))
            qadr, vadr = int(self.m.jnt_qposadr[jid]), int(self.m.jnt_dofadr[jid])
            self.objects[name] = FreeObject(name, self.d.qpos[qadr:qadr + 7].copy(), qadr, vadr, geoms,
                np.stack([self.m.geom_contype[geoms], self.m.geom_conaffinity[geoms]]))
        self.robot_geoms = np.array([i for i, n in enumerate(self.env.sim.model.geom_names)
                                    if n and (n.startswith("robot") or n.startswith("gripper"))], dtype=int)
        self.gripper_geoms = set(i for i, n in enumerate(self.env.sim.model.geom_names)
                                 if n and ("gripper" in n or "finger" in n))
        self.s = self.stage_count = 0
        if self.target not in self.objects and self.condition in ("onset_arrival", "mid_arrival", "mid_shift", "remove_return"):
            self.invalid = f"no eligible free object: {self.target}"
        if self.variant == "remove":
            self._distractor(True)
        self._apply()
        mujoco.mj_forward(self.m, self.d)

    def notify_stage(self, name):
        self.stage_count += 1
        self.events.append({"step": self.s, "stage": name})

    def _pos(self, name):
        obj = self.objects[name]
        return self.d.qpos[obj.qadr:obj.qadr + 3]

    def _near(self, point, distance=.10):
        return bool(len(self.robot_geoms) and np.min(np.linalg.norm(self.d.geom_xpos[self.robot_geoms] - point, axis=1)) < distance)

    def _manipulated(self, name):
        obj = self.objects[name]
        if self._pos(name)[2] > obj.pose[2] + .025:
            return True
        for contact in self.d.contact[:self.d.ncon]:
            if contact.dist > .002:
                continue
            if ((contact.geom1 in obj.geoms and contact.geom2 in self.gripper_geoms)
                    or (contact.geom2 in obj.geoms and contact.geom1 in self.gripper_geoms)):
                return True
        return False

    def _visible(self, point, object_geoms=()):
        import mujoco
        for camera in ("agentview", "robot0_eye_in_hand"):
            cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if cid < 0:
                continue
            origin = self.d.cam_xpos[cid]
            relative = self.d.cam_xmat[cid].reshape(3, 3).T @ (point - origin)
            depth = -relative[2]
            bound = depth * np.tan(np.deg2rad(self.m.cam_fovy[cid] / 2))
            if depth <= 0 or abs(relative[0]) > bound * 4 / 3 or abs(relative[1]) > bound:
                continue
            direction = point - origin
            length = np.linalg.norm(direction)
            hit = np.array([-1], dtype=np.int32)
            ray_distance = mujoco.mj_ray(self.m, self.d, origin, direction / length,
                                         np.ones(6, dtype=np.uint8), 1, -1, hit)
            if hit[0] in object_geoms or ray_distance < 0 or ray_distance >= length - .04:
                return True
        return False

    def _begin(self):
        self.started = self.s
        self.events.append({"step": self.s, "event": "start"})

    def _visible_object(self, obj):
        for geom in obj.geoms:
            center = self.d.geom_xpos[geom]
            radius = min(float(self.m.geom_rbound[geom]), .05)
            for offset in ((0, 0, 0), (radius * .6, 0, radius * .5),
                           (-radius * .6, 0, radius * .5), (0, radius * .6, radius * .5),
                           (0, -radius * .6, radius * .5)):
                if self._visible(center + offset, obj.geoms):
                    return True
        return False

    def _finish(self):
        self.ended = self.s
        self.active_pose = None
        self._mocap("ri_hand", [-3, 0, 2])
        self.events.append({"step": self.s, "event": "end"})

    def _tick(self):
        if self.condition == "clean" or self.invalid:
            return
        if self.s < 10:
            return
        if self.s == 10:
            for obj in self.objects.values():
                obj.pose[:] = self.d.qpos[obj.qadr:obj.qadr + 7]
            if self.condition in ("onset_arrival", "mid_arrival"):
                self.active_pose = self.objects[self.target].pose.copy()
                self.active_pose[1] += .9 if self.condition == "onset_arrival" else 1.8
        c = self.condition
        first, later = TARGETS[self.task]
        manipulation_started = self.stage_count > 0 if self.task in (4, 5) else self._manipulated(first)
        if self.s >= 10 and manipulation_started and self.first_manipulation is None:
            self.first_manipulation = self.s
        if later in self.objects and self._manipulated(later) and self.active_pose is None:
            self.target_manipulation = self.target_manipulation or self.s
        if c == "onset_arrival":
            if self.s == 10:
                self._begin()
            if self.started is not None and self.ended is None:
                if self.first_manipulation is not None and self.s < 50:
                    self.invalid = "onset did not finish before first manipulation"
                alpha = np.clip((self.s - 10) / 40, 0, 1)
                self.active_pose = self.objects[self.target].pose.copy()
                self.active_pose[1] += .9 * (1 - self._ease(alpha))
                if self.s >= 50:
                    self._put(self.active_pose)
                    self._finish()
        elif c == "mid_arrival":
            required_soon = self._near(self.objects[self.target].pose[:3], .18) or self.stage_count > 0
            if self.started is None and self.first_manipulation is not None and (self.s >= 108 or required_soon):
                self._begin()
                self.duration = max(1, min(12, 120 - self.s))
            if self.started is not None and self.ended is None:
                alpha = np.clip((self.s - self.started) / self.duration, 0, 1)
                self.active_pose = self.objects[self.target].pose.copy()
                self.active_pose[1] += 1.8 * (1 - self._ease(alpha))
                if alpha >= 1:
                    if self._near(self.active_pose[:3], .045):
                        self.invalid = "arrival missed manipulation boundary"
                    self._put(self.active_pose)
                    self._finish()
        elif c in ("mid_shift", "remove_return"):
            if c == "remove_return" and self.started is None and self.s > 152:
                self.invalid = "no eligible removal/return window before recorded step 160"
                return
            if self.started is None and self.s >= 80 and self.first_manipulation is not None:
                if self.target_manipulation is not None:
                    self.invalid = "target already manipulated before intervention"
                    return
                obj = self.objects[self.target]
                if self._near(self._pos(self.target), .12):
                    return
                if np.linalg.norm(self._pos(self.target) - obj.pose[:3]) > .025:
                    return
                self.saved = self.d.qpos[obj.qadr:obj.qadr + 7].copy()
                self._begin()
                if c == "mid_shift":
                    self.destination = self._shift_destination()
                    if self.destination is None:
                        self.invalid = "no collision-free visible shift"
                        return
            if self.started is not None and self.ended is None:
                self.active_pose = self.saved.copy()
                if c == "mid_shift":
                    alpha = min(1, (self.s - self.started) / 8)
                    self.active_pose[:3] += self._ease(alpha) * (self.destination[:3] - self.saved[:3])
                    if self._near(self.active_pose[:3], .045):
                        self.invalid = "shift overlaps target manipulation"
                    if alpha >= 1:
                        self._put(self.active_pose)
                        self._finish()
                else:
                    age = self.s - self.started
                    deadline = min(self.started + 80, 160)
                    returning = self._near(self.saved[:3], .18) or self.s >= deadline - 8
                    if returning and not hasattr(self, "return_at"):
                        self.return_at = self.s
                    phase = self._ease(min(1, age / 8))
                    if hasattr(self, "return_at"):
                        phase *= 1 - self._ease(min(1, (self.s - self.return_at) / 8))
                    self.active_pose[:3] += [0, .9 * phase, .15 * phase]
                    if hasattr(self, "return_at") and self.s >= self.return_at + 8:
                        if self._near(self.saved[:3], .045):
                            self.invalid = "return overlaps target manipulation"
                        self._put(self.saved)
                        self._finish()
        else:
            self._tick_distractor()
        self._apply()
        import mujoco
        mujoco.mj_forward(self.m, self.d)
        task_object = c in ("onset_arrival", "mid_arrival", "mid_shift", "remove_return")
        if self.started is not None and (self.ended is None or (task_object and self.s <= self.ended + 4)):
            if c in ("camera_shake", "light_change"):
                visible = True
            elif c == "tv_on":
                geoms = [self.env.sim.model.geom_name2id(f"ri_stripe_{i}") for i in range(8)]
                visible = self._visible(np.array([-1.727, .625, .577]), geoms)
            elif c == "object_passby":
                bid = self.env.sim.model.body_name2id("ri_person")
                geoms = np.flatnonzero(self.m.geom_bodyid == bid)
                visible = any(self._visible(self.d.geom_xpos[g], geoms) for g in geoms)
            elif c in ("unrelated_object", "target_like_object"):
                bid = self.env.sim.model.body_name2id("ri_object")
                geoms = np.flatnonzero(self.m.geom_bodyid == bid)
                visible = self._visible(self.d.xpos[bid], geoms)
            else:
                visible = self._visible_object(self.objects[self.target])
            self.visible_steps += int(visible)

    @staticmethod
    def _ease(x):
        return x * x * x * (10 + x * (-15 + 6 * x))

    def _shift_destination(self):
        import mujoco
        obj = self.objects[self.target]
        distance = .025 if ((self.seed - 50) // 2) % 2 == 0 else .04
        options = []
        for delta in ([distance, 0, 0], [-distance, 0, 0], [0, distance, 0], [0, -distance, 0]):
            pose = self.saved.copy()
            pose[:3] += delta
            if self._near(pose[:3], .12):
                continue
            self._put(pose)
            mujoco.mj_forward(self.m, self.d)
            collisions = [contact for contact in self.d.contact[:self.d.ncon]
                          if (contact.geom1 in obj.geoms) != (contact.geom2 in obj.geoms)]
            bad = any(contact.dist < -.003 for contact in collisions)
            other = [np.linalg.norm(pose[:2] - self._pos(n)[:2]) for n in self.objects if n != self.target]
            if not bad and min(other, default=1) > .065 and self._visible(pose[:3], obj.geoms):
                options.append((min(other, default=1), pose))
        self._put(self.saved)
        mujoco.mj_forward(self.m, self.d)
        return max(options, key=lambda p: p[0])[1] if options else None

    def _put(self, pose):
        obj = self.objects[self.target]
        self.d.qpos[obj.qadr:obj.qadr + 7] = pose
        self.d.qvel[obj.vadr:obj.vadr + 6] = 0

    def _mocap(self, name, pos):
        bid = self.env.sim.model.body_name2id(name)
        self.d.mocap_pos[self.m.body_mocapid[bid]] = pos

    def _distractor(self, visible):
        if not hasattr(self, "distractor_position"):
            task_positions = [o.pose[:3] for o in self.objects.values()]
            candidates = [np.array([x, y, .46]) for x in (-.28, .25) for y in (-.62, .72)]
            self.distractor_position = max(candidates, key=lambda p: min(np.linalg.norm(p[:2] - q[:2]) for q in task_positions))
        self._mocap("ri_object", self.distractor_position if visible else [-3, 0, 2])
        like = self.condition == "target_like_object"
        if like:
            palette = {"butter": (.9, .8, .25), "cream_cheese": (.92, .9, .75),
                       "chocolate_pudding": (.4, .24, .13), "tomato_sauce": (.8, .12, .10),
                       "cookies": (.65, .4, .2), "popcorn": (.9, .6, .2), "milk": (.8, .85, .95)}
            self.m.geom_rgba[self.env.sim.model.geom_name2id("ri_box"), :3] = palette.get(self.target.removesuffix("_1"), (.7, .6, .3))
        for name, enabled in (("ri_sphere", not like), ("ri_box", like), ("ri_cross_h", like), ("ri_cross_v", like)):
            gid = self.env.sim.model.geom_name2id(name)
            self.m.geom_rgba[gid, 3] = float(enabled)

    def _tick_distractor(self):
        c, s = self.condition, self.s
        if self.started is None and s >= 40:
            self._begin()
        if self.started is None or self.ended is not None:
            return
        if c == "tv_on":
            for i in range(8):
                gid = self.env.sim.model.geom_name2id(f"ri_stripe_{i}")
                self.m.geom_rgba[gid] = [*((np.array([i * 31, i * 59, i * 83]) + s * 9) % 255 / 255), 1]
        elif c == "object_passby":
            self._mocap("ri_person", [-1.25, -1.45 + 2.9 * min(1, (s - 40) / 240), -.15])
        elif c in ("unrelated_object", "target_like_object"):
            self._distractor(s < 160 if self.variant == "remove" else True)
        elif c == "camera_shake":
            cid = self.env.sim.model.camera_name2id("agentview")
            phase = np.sin(2 * np.pi * (s - 40 + .25) / 3)
            axes = self.d.cam_xmat[cid].reshape(3, 3)
            self.m.cam_pos[cid] = self.camera_pos[cid] + phase * (axes[:, 1] * .010 + axes[:, 2] * .008)
        elif c == "light_change":
            multiplier = 1.9 if self.variant == "bright" else .35
            self.m.light_ambient[:] = self.ambient * multiplier
            self.m.light_diffuse[:] = self.diffuse * multiplier
        end = 43 if c in ("camera_shake", "light_change") else (160 if self.variant == "remove" else 280)
        if s >= end:
            self.m.cam_pos[:] = self.camera_pos
            self.m.light_ambient[:], self.m.light_diffuse[:] = self.ambient, self.diffuse
            if c == "object_passby":
                self._mocap("ri_person", [-3, 0, 2])
            elif c == "tv_on":
                for i in range(8):
                    self.m.geom_rgba[self.env.sim.model.geom_name2id(f"ri_stripe_{i}"), 3] = 0
            elif c in ("unrelated_object", "target_like_object"):
                self._distractor(False)
            self._finish()

    def _apply(self):
        if self.target in self.objects:
            obj = self.objects[self.target]
            if self.active_pose is not None and not self.invalid:
                self._put(self.active_pose)
                self.m.geom_contype[obj.geoms] = self.m.geom_conaffinity[obj.geoms] = 0
                if self.style == "handin" and self.started is not None and self.ended is None:
                    self._mocap("ri_hand", self.active_pose[:3] + [0, 0, .06])
            else:
                self.m.geom_contype[obj.geoms], self.m.geom_conaffinity[obj.geoms] = obj.collision

    def report(self):
        valid = self.condition == "clean" or (self.started is not None and self.ended is not None
            and self.invalid is None and self.visible_steps > 0)
        return {"condition": self.condition, "variant": self.variant, "style": self.style,
                "target": self.target, "start": self.started, "end": self.ended,
                "first_manipulation": self.first_manipulation, "target_manipulation": self.target_manipulation,
                "valid": valid, "invalid_reason": self.invalid or (None if valid else
                    "event was not camera-visible" if self.ended is not None else "event not completed"),
                "visible_control_steps": self.visible_steps,
                "paper_default_timing_met": (self.ended is not None and self.ended <= 120)
                    if self.condition == "mid_arrival" else True,
                "initial_scene_sha256": self.scene_hash, "events": self.events}
