"""No History, Shuffled History and OracleMem (supplement Table 4)."""

import numpy as np
from .scene import array
from .rimem.evidence import MemoryKey, grid_keys


def patch_permutation(keys, now, rng):
    groups = {}
    for i, key in enumerate(keys):
        if key.frames and max(key.frames) < now:
            groups.setdefault((key.view, key.box), []).append(i)
    permutation = np.arange(len(keys))
    for ids in groups.values():
        permutation[ids] = rng.permutation(ids)
    return permutation


def privileged_ids(env):
    u = env.unwrapped
    ids = []
    for name, actor in u.scene.actors.items():
        if name not in ("ground", "table-workspace") and not name.startswith("ri_"):
            ids.extend(array(actor.per_scene_id).reshape(-1).tolist())
    for art in u.scene.articulations.values():
        for link in art.links:
            ids.extend(array(link.per_scene_id).reshape(-1).tolist())
    return ids


class Control:
    def __init__(self, kind):
        if kind not in ("no-history", "shuffled-history", "oracle-mem"):
            raise ValueError(kind)
        self.kind = kind

    def reset(self, seed, env):
        self.rng = np.random.default_rng(seed)
        self.masks = []
        self.now = -1
        self.ids = privileged_ids(env) if self.kind == "oracle-mem" else []

    def observe(self, obs):
        n = len(obs["front_rgb_list"])
        if self.kind == "oracle-mem":
            raw = obs.get("maniskill_obs")
            raw = raw if isinstance(raw, list) else [raw]
            if len(raw) != n or any(r is None for r in raw):
                raise ValueError(
                    "OracleMem requires one native segmentation observation per frame"
                )
            for r in raw:
                seg = array(r["sensor_data"]["base_camera"]["segmentation"]).squeeze()
                if seg.ndim != 2:
                    raise ValueError("Expected native actor-ID segmentation")
                self.masks.append(np.isin(seg, self.ids))
        self.now += n

    def keep(self, key):
        if not key.frames:
            return False
        if max(key.frames) >= self.now:
            return True
        x0, y0, x1, y1 = key.box
        masks = [self.masks[t] for t in key.frames]
        return any(
            mask[
                int(y0 * mask.shape[0]) : max(
                    int(y1 * mask.shape[0]), int(y0 * mask.shape[0]) + 1
                ),
                int(x0 * mask.shape[1]) : max(
                    int(x1 * mask.shape[1]), int(x0 * mask.shape[1]) + 1
                ),
            ].any()
            for mask in masks
        )


def attach_mme(policy, control):
    import jax.numpy as jnp

    mem = policy.mem_buffer
    native_prepare = policy._prepare_history
    native_frames, native_tokens = (
        mem.get_frame_sampling_indices,
        mem.prepare_token_dropping,
    )
    keys = []

    def frames(step, budget, tokens):
        ids = native_frames(step, budget, tokens)
        keys[:] = grid_keys(ids, int(tokens**0.5), budget=budget)
        return ids

    def tokens(step, budget, gather, *a, **kw):
        ids = mem.filter_token_dropping_indices(
            mem.get_token_dropping_indices(), step, budget
        )
        keys[:] = [
            MemoryKey(
                (int(t),), (p % 8 / 8, p // 8 / 8, (p % 8 + 1) / 8, (p // 8 + 1) / 8)
            )
            for t, v, p in ids
        ]
        keys.extend([MemoryKey((), (0, 0, 1, 1))] * (budget - len(keys)))
        return native_tokens(step, budget, gather, *a, kept_indices=ids, **kw)

    mem.get_frame_sampling_indices, mem.prepare_token_dropping = frames, tokens

    def prepare(inputs):
        data = native_prepare(inputs)
        if control.kind == "no-history":
            data["static_mask"] = np.zeros_like(data["static_mask"], dtype=bool)
            policy._sample_kwargs.update(
                ri_bias=jnp.zeros((1, len(keys))), ri_gate=jnp.zeros(1)
            )
        elif control.kind == "shuffled-history":
            perm = patch_permutation(keys, control.now, control.rng)
            for name in ("static_image_emb", "static_state_emb"):
                data[name] = np.asarray(data[name])[perm]
        elif control.kind == "oracle-mem":
            keep = np.array([control.keep(k) for k in keys], bool)
            data["static_mask"] = np.asarray(data["static_mask"]) & keep
            policy._sample_kwargs.update(
                ri_bias=jnp.zeros((1, len(keys))),
                ri_gate=jnp.array([float(keep.any())]),
            )
        return data

    policy._prepare_history = prepare

    def detach():
        policy._prepare_history = native_prepare
        mem.get_frame_sampling_indices, mem.prepare_token_dropping = (
            native_frames,
            native_tokens,
        )
        policy._sample_kwargs.pop("ri_bias", None)
        policy._sample_kwargs.pop("ri_gate", None)

    return detach


def attach_simple(buffer, control):
    positions, pixels = buffer._history_positions, buffer._preprocess_patch_pixels
    video = getattr(buffer, "_history_video", None)

    def no_positions(view):
        n = len(buffer._buffers[view]) - 1
        return [n, n]

    def oracle_video(view):
        clip = video(view).copy()
        base = buffer._frames_seen[view] - len(buffer._buffers[view])
        for i, pos in enumerate(positions(view)):
            frame = base + pos
            if frame < control.now:
                clip[i][~control.masks[frame]] = 128
        return clip

    def oracle_pixels(key):
        view, fa, fb = key
        camera = next(k for k in buffer.history_image_keys if k.split(".")[-1] == view)
        saved = {}
        base = buffer._frames_seen[camera] - len(buffer._buffers[camera])
        for frame in (fa, fb):
            if frame < control.now and frame not in saved:
                idx = frame - base
                image = buffer._buffers[camera][idx]
                saved[frame] = image
                altered = image.copy()
                altered[~control.masks[frame]] = 128
                buffer._buffers[camera][idx] = altered
        try:
            return pixels(key)
        finally:
            for frame, image in saved.items():
                buffer._buffers[camera][frame - base] = image

    if control.kind == "no-history":
        buffer._history_positions = no_positions
    elif control.kind == "oracle-mem":
        buffer._history_video = oracle_video
        buffer._preprocess_patch_pixels = oracle_pixels

    def detach():
        buffer._history_positions, buffer._preprocess_patch_pixels = positions, pixels
        if video is not None:
            buffer._history_video = video

    return detach


def simple_processed(processed, control):
    if control.kind != "shuffled-history":
        return processed
    keys = processed["video_patch_keys"]
    order = list(control.rng.permutation(max(0, len(keys) - 1))) + (
        [len(keys) - 1] if keys else []
    )
    return {**processed, "video_patch_keys": [keys[i] for i in order]}
