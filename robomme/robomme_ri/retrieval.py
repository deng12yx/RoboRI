"""RoboMME-Interference comparator: cosine segments on initial video history."""

import numpy as np


def select_segments(embeddings, query, threshold=0.9, min_frames=4):
    if not len(embeddings):
        return []
    x = np.asarray(embeddings, np.float32)
    q = np.asarray(query, np.float32)
    if (
        x.ndim != 2
        or q.shape != (x.shape[1],)
        or not np.isfinite(x).all()
        or not np.isfinite(q).all()
    ):
        raise ValueError("Invalid normalized retrieval embeddings")
    cuts = (
        [0]
        + (np.flatnonzero(np.sum(x[1:] * x[:-1], axis=1) < threshold) + 1).tolist()
        + [len(x)]
    )
    segments = [(a, b) for a, b in zip(cuts, cuts[1:]) if b - a >= min_frames]
    if not segments:
        return list(range(len(x)))
    a, b = max(segments, key=lambda ab: float(np.max(x[ab[0] : ab[1]] @ q)))
    return list(range(a, b))


class SigLIPRetrieval:
    def __init__(self, model_path, device="cuda:0"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = (
            AutoModel.from_pretrained(model_path, local_files_only=True)
            .to(device)
            .eval()
        )

    def encode(self, frames):
        from PIL import Image

        chunks = []
        with self.torch.inference_mode():
            for i in range(0, len(frames), 8):
                batch = self.processor(
                    images=[Image.fromarray(f) for f in frames[i : i + 8]],
                    return_tensors="pt",
                ).to(self.device)
                feats = self.model.get_image_features(**batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                chunks.append(feats.half().float().cpu().numpy())
        return np.concatenate(chunks)

    def __call__(self, history, current):
        if not history:
            return []
        features = self.encode([*history, current])
        return select_segments(features[:-1], features[-1])
