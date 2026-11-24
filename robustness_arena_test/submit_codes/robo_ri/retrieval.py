"""Visual segment retrieval adapted from robomme-interference, revision 3536f63.

Upstream: https://github.com/SoumilRathi/robomme-interference
See THIRD_PARTY.md for attribution and the streaming adaptation.
"""
from __future__ import annotations

import json
import weakref

import numpy as np


def select_segment(embeddings, query, threshold=.923, min_frames=4):
    embeddings, query = np.asarray(embeddings, np.float32), np.asarray(query, np.float32)
    if embeddings.ndim != 2 or query.shape != embeddings.shape[1:]:
        raise ValueError("Expected [frames, dim] embeddings and a [dim] query")
    if not np.isfinite(embeddings).all() or not np.isfinite(query).all():
        raise ValueError("Nonfinite retrieval features")
    count = len(embeddings)
    adjacent = np.sum(embeddings[:-1] * embeddings[1:], axis=-1)
    cuts = [0, *(np.flatnonzero(adjacent < threshold) + 1).tolist(), count]
    segments = [(lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:]) if hi - lo >= min_frames]
    if not segments:
        return list(range(count))
    best = max(segments, key=lambda pair: float(np.max(embeddings[pair[0]:pair[1]] @ query)))
    return list(range(*best))


class Encoder:
    def __init__(self, checkpoint, device="cuda:0"):
        import open_clip
        import torch
        self.torch, self.device = torch, device
        configuration = json.loads((checkpoint / "open_clip_config.json").read_text())["preprocess_cfg"]
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-SO400M-14-SigLIP-384", pretrained=str(checkpoint / "open_clip_model.safetensors"),
            precision="fp32", device="cpu", pretrained_hf=False,
            image_mean=configuration["mean"], image_std=configuration["std"],
            image_interpolation=configuration.get("interpolation", "bicubic"),
            image_resize_mode=configuration.get("resize_mode", "squash"))
        self.model = model.visual.to(device).eval().requires_grad_(False)

    def images(self, frames):
        from PIL import Image
        result = []
        with self.torch.inference_mode():
            for start in range(0, len(frames), 2):
                pixels = self.torch.stack([self.preprocess(Image.fromarray(np.asarray(f, np.uint8)))
                                          for f in frames[start:start + 2]]).to(self.device)
                features = self.model(pixels)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                result.append(features.half().cpu().numpy())
        return np.concatenate(result)


class FrameBank:
    def __init__(self, encoder):
        self.encoder, self.features, self.trace = encoder, {}, []

    def add(self, frames, indices):
        if any(i in self.features for i in indices):
            raise ValueError("Frame index reused within an episode")
        self.features.update(zip(indices, self.encoder.images(frames)))

    def bind(self, buffer):
        reference = weakref.ref(buffer)
        def select(current, budget, per_image):
            memory = reference()
            if memory is None:
                raise RuntimeError("Memory buffer already released")
            indices = sorted(i for i in memory._history_feats if i < current and i in self.features)
            capacity = budget // (per_image * memory.num_views)
            if not indices:
                return [current] if capacity else []
            positions = select_segment(np.stack([self.features[i] for i in indices]), self.features[current])
            selected = [indices[i] for i in positions]
            if len(selected) > capacity:
                selected = [selected[i] for i in np.linspace(0, len(selected) - 1, capacity, dtype=int)]
            self.trace.append({"query": current, "selected": selected})
            return selected
        buffer.get_frame_sampling_indices = select
