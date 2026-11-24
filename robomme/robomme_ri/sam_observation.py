"""SAM2Act RGB-D/point-cloud input conversion, 128 px native input."""

import numpy as np
from .scene import array


def convert(front, depth, K, E, wrist, wdepth, wK, wE, qpos, p, q):
    import cv2

    result = {}
    for rgb, d, ki, ex, rgbkey, depthkey, cam in (
        (front, depth, K, E, "image", "base_camera_depth", "base_camera"),
        (wrist, wdepth, wK, wE, "wrist_image", "wrist_camera_depth", "wrist_camera"),
    ):
        rgb = np.asarray(rgb, np.uint8)
        h, w = rgb.shape[:2]
        result[rgbkey] = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
        d = np.asarray(d, np.float32).squeeze() / 1000.0
        result[depthkey] = cv2.resize(d, (128, 128), interpolation=cv2.INTER_LINEAR)
        k = np.array(ki, np.float64).reshape(3, 3).copy()
        k[0] *= 128 / w
        k[1] *= 128 / h
        result[cam + "_intrinsic_opencv"] = k
        result[cam + "_extrinsic_opencv"] = np.asarray(ex, np.float64)
    qpos = np.asarray(qpos).reshape(-1)
    qw, qx, qy, qz = np.asarray(q).reshape(4)
    result.update(
        joint_positions=qpos[:7],
        gripper_joint_positions=qpos[7:9],
        gripper_open=float(qpos[7:9].sum() > 0.002),
        robot_endeffector_p=np.asarray(p).reshape(3),
        robot_endeffector_q=np.array([qx, qy, qz, qw]),
        ignore_collisions=1,
        misc={},
    )
    return result


def from_lists(env, obs, info, i):
    agent = env.unwrapped.agent
    return convert(
        obs["front_rgb_list"][i],
        obs["front_depth_list"][i],
        info["front_camera_intrinsic"],
        obs["front_camera_extrinsic_list"][i],
        obs["wrist_rgb_list"][i],
        obs["wrist_depth_list"][i],
        info["wrist_camera_intrinsic"],
        obs["wrist_camera_extrinsic_list"][i],
        np.r_[obs["joint_state_list"][i], obs["gripper_state_list"][i]],
        array(agent.tcp.pose.p),
        array(agent.tcp.pose.q),
    )


def from_raw(env):
    raw = env.unwrapped.get_obs()
    sd, sp = raw["sensor_data"], raw["sensor_param"]
    a = env.unwrapped.agent
    drop = lambda x: array(x)[0]
    return convert(
        drop(sd["base_camera"]["rgb"]),
        drop(sd["base_camera"]["depth"]),
        drop(sp["base_camera"]["intrinsic_cv"]),
        drop(sp["base_camera"]["extrinsic_cv"]),
        drop(sd["hand_camera"]["rgb"]),
        drop(sd["hand_camera"]["depth"]),
        drop(sp["hand_camera"]["intrinsic_cv"]),
        drop(sp["hand_camera"]["extrinsic_cv"]),
        drop(raw["agent"]["qpos"]),
        array(a.tcp.pose.p),
        array(a.tcp.pose.q),
    )
