"""Adapters for official baseline services; each model keeps its own environment.

These are inference clients, not reimplementations of pretrained models. Start
the matching official model server before evaluation. All endpoints are explicit.
"""

import base64
import json
import numpy as np
from .baselines import METHODS
from .policies import frames, instruction


def encode(o):
    if isinstance(o, (np.ndarray, np.generic)):
        a = np.ascontiguousarray(o)
        return {
            "__numpy__": base64.b64encode(a.tobytes()).decode(),
            "dtype": a.dtype.str,
            "shape": list(a.shape),
        }
    raise TypeError(type(o).__name__)


def decode(o):
    if "__numpy__" in o:
        return np.frombuffer(
            base64.b64decode(o["__numpy__"]), dtype=np.dtype(o["dtype"])
        ).reshape(o["shape"])
    return o


class HTTP:
    def __init__(self, url):
        import requests

        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False

    def call(self, path, payload):
        r = self.session.post(
            self.url + path,
            data=json.dumps(payload, default=encode),
            headers={"Content-Type": "application/json"},
            timeout=900,
        )
        r.raise_for_status()
        return json.loads(r.text, object_hook=decode)

    def close(self):
        self.session.close()


class WebSocket:
    def __init__(self, url):
        from websockets.sync.client import connect
        from openpi_client import msgpack_numpy

        self.codec = msgpack_numpy
        self.ws = connect(url, compression=None, max_size=None, ping_interval=None)
        self.metadata = self.codec.unpackb(self.ws.recv())

    def call(self, payload):
        self.ws.send(self.codec.packb(payload))
        r = self.ws.recv()
        if isinstance(r, str):
            raise RuntimeError(r)
        return self.codec.unpackb(r)

    def close(self):
        self.ws.close()


class OFTPolicy:
    action_space = "joint_angle"

    def __init__(self, method, cfg):
        self.http = HTTP(cfg["endpoint"])
        self.method = METHODS[method]
        self.sg = None
        if self.method.subgoals == "qwen":
            from .predictors import create_predictor

            self.sg = create_predictor(self.method.native, cfg)
        self.metadata = {
            "native": self.method.native,
            "endpoint": cfg["endpoint"],
            "subgoals": self.method.subgoals,
        }

    def reset(self, spec, obs, info, env):
        self.goal, self.count, self.pending = instruction(info), 0, []
        fs = frames(obs)
        self.http.call(
            "/reset",
            {
                "images": [f for f, w, s in fs[:-1]],
                "states": [s for f, w, s in fs[:-1]],
            },
        )
        if self.sg:
            self.sg.reset(spec, obs, self.goal)
        self.current = fs[-1]

    def observe(self, obs):
        self.pending.extend(frames(obs))
        self.current = self.pending[-1]
        self.count += len(frames(obs))
        if self.sg:
            self.sg.observe(obs, self.count)

    def act(self, obs, info):
        # Native OFT /act itself adds the current frame. Add all intervening
        # frames except the last one, exactly as the official streaming client.
        for f, w, s in self.pending[:-1]:
            self.http.call("/add_frame", {"image": f, "state": s})
        self.pending.clear()
        f, w, s = self.current
        prompt = self.goal.lower()
        if self.sg:
            prompt = (
                f"Task: {prompt}\nCurrent subgoal: {self.sg.predict(f, self.count)}"
            )
        r = self.http.call(
            "/act",
            {"full_image": f, "wrist_image": w, "state": s, "instruction": prompt},
        )
        return np.asarray(
            r.get("actions", r.get("action_chunk")) if isinstance(r, dict) else r,
            np.float32,
        )

    def close(self):
        self.http.close()
        if self.sg:
            self.sg.close()


class SmolPolicy:
    action_space = "joint_angle"

    def __init__(self, cfg):
        self.http = HTTP(cfg["endpoint"])
        self.metadata = {"native": "smolvla", "endpoint": cfg["endpoint"]}

    def reset(self, spec, obs, info, env):
        self.http.call("/reset", {"seed": spec.model_seed})
        self.goal = instruction(info)
        self.observe(obs)

    def observe(self, obs):
        self.current = frames(obs)[-1]

    def act(self, obs, info):
        f, w, s = self.current
        return np.asarray(
            self.http.call(
                "/act", {"front": f, "wrist": w, "state": s, "task": self.goal}
            )["actions"],
            np.float32,
        )

    def close(self):
        self.http.close()


class ObservingWS:
    """Released observation/RPC wrappers for Diffusion Policy and MemoryVLA."""

    def __init__(self, method, cfg):
        self.method = method
        self.ws = WebSocket(cfg["endpoint"])
        self.metadata = {"native": method, **self.ws.metadata}
        self.action_space = "ee_pose" if method == "memoryvla" else "joint_angle"
        self.exec_horizon = 8

    def reset(self, spec, obs, info, env):
        self.ws.call({"op": "reset", "seed": spec.model_seed})
        self.goal = instruction(info)
        self.n = 0
        self.observe(obs, initial=True)

    def observe(self, obs, initial=False):
        fs = frames(obs)
        self.current = fs[-1]
        if self.method == "memoryvla":
            payload = []
            for i, (f, w, s) in enumerate(fs):
                payload.append(
                    dict(
                        front=f,
                        wrist=w,
                        index=self.n,
                        phase="ENCODE" if initial and i < len(fs) - 1 else "EXEC",
                        passive=initial and i < len(fs) - 1,
                        passive_offset=i,
                    )
                )
                self.n += 1
        else:
            payload = fs
        self.ws.call({"op": "observe", "frames": payload})

    def act(self, obs, info):
        req = {"op": "infer", "prompt": self.goal}
        if self.method != "memoryvla":
            f, w, s = self.current
            req.update(front=f, wrist=w, state=s)
        return np.asarray(self.ws.call(req)["actions"], np.float32)

    def convert_action(self, action, obs):
        if self.method != "memoryvla":
            return action
        ee = np.asarray(obs["eef_state_list"][-1], np.float64).reshape(-1)
        rpy = ee[3:6].copy()
        rpy[0] += 2 * np.pi
        return np.r_[ee[:3] + action[:3], rpy + action[3:6], action[6]]

    def close(self):
        self.ws.close()


class HAMLETPolicy:
    action_space = "joint_angle"

    def __init__(self, cfg):
        from gr00t.policy.server_client import PolicyClient
        from gr00t.eval.sim.robomme import run_robomme_rollout

        self.native = run_robomme_rollout
        self.client = PolicyClient(host=cfg.get("host", "127.0.0.1"), port=cfg["port"])
        self.metadata = {"native": "HAMLET", "port": cfg["port"]}

    def reset(self, spec, obs, info, env):
        self.goal = instruction(info)
        self.session = spec.key()
        self.native._prime_hamlet_memory(
            self.client, obs, self.session, self.goal, 4, 16
        )
        self.first = len(obs["front_rgb_list"]) <= 1
        self.observe(obs)

    def observe(self, obs):
        self.current = self.native._build_step_obs(obs, -1, self.goal)

    def act(self, obs, info):
        actions, _ = self.client.get_action(
            self.current,
            options={"session_ids": [self.session], "reset_memory": [self.first]},
        )
        self.first = False
        jp, gp = (
            np.asarray(actions["action.joint_position"]),
            np.asarray(actions["action.gripper_close"]),
        )
        if jp.ndim == 3:
            jp, gp = jp[0], gp[0]
        return np.concatenate((jp, gp.reshape(len(jp), -1)), axis=-1)

    def close(self):
        pass


class SAMPolicy:
    action_space = "waypoint"
    exec_horizon = 1
    max_control_steps = 100000
    max_decisions = 40
    env_kwargs = dict(
        include_front_depth=True,
        include_wrist_depth=True,
        include_front_camera_extrinsic=True,
        include_wrist_camera_extrinsic=True,
        include_front_camera_intrinsic=True,
        include_wrist_camera_intrinsic=True,
    )

    def __init__(self, method, cfg):
        self.method = method
        self.ws = WebSocket(cfg["endpoint"])
        self.metadata = {"native": method, "max_waypoints": 40, **self.ws.metadata}

    def reset(self, spec, obs, info, env):
        from .sam_observation import from_lists

        self.env, self.goal, self.count = env, instruction(info), 0
        self.ws.call({"reset": True})
        if self.method == "sam2act-plus":
            n = len(obs["front_rgb_list"])
            ids = (
                np.unique(np.rint(np.linspace(0, n - 1, min(16, n))).astype(int))
                if n
                else []
            )
            batch = [
                dict(
                    obs_obj=from_lists(env, obs, info, int(i)),
                    curr_idx=j,
                    lang_goal=self.goal,
                    episode_length=40,
                )
                for j, i in enumerate(ids)
            ]
            if batch:
                self.ws.call(
                    {
                        "add_buffer": True,
                        "frames": batch,
                        "lang_goal": self.goal,
                        "episode_length": 40,
                    }
                )

    def observe(self, obs):
        pass

    def act(self, obs, info):
        from .sam_observation import from_raw
        from scipy.spatial.transform import Rotation

        r = self.ws.call(
            {
                "obs_obj": from_raw(self.env),
                "curr_idx": self.count,
                "lang_goal": self.goal,
                "episode_length": 40,
            }
        )
        self.count += 1
        a = np.asarray(r["actions"][0], np.float64)
        return np.r_[
            a[:3],
            Rotation.from_quat(a[3:7]).as_euler("xyz"),
            1.0 if a[7] > 0.5 else -1.0,
        ][None]

    def close(self):
        self.ws.close()


def create_official(method, cfg):
    if method.startswith("oft-") or method == "openvla-oft":
        return OFTPolicy(method, cfg)
    if method == "smolvla":
        return SmolPolicy(cfg)
    if method in ("memoryvla", "diffusion-policy"):
        return ObservingWS(method, cfg)
    if method == "hamlet":
        return HAMLETPolicy(cfg)
    if method in ("sam2act", "sam2act-plus"):
        return SAMPolicy(method, cfg)
    raise ValueError(f"No official service adapter for {method}")
