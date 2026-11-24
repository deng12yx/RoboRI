"""Evidence scores and retained-key alignment: supplement Eqs. 2–7."""

from collections import OrderedDict
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Region:
    frame: int
    label: str
    box: tuple[float, float, float, float]
    confidence: float
    view: str = "front"
    distractor: bool = False

    def __post_init__(self):
        x0, y0, x1, y1 = self.box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 and 0 <= self.confidence <= 1):
            raise ValueError("Expected normalized nonempty box and confidence in [0,1]")


@dataclass(frozen=True)
class MemoryKey:
    # A temporal merge has multiple source frames. Padding has no source frames.
    frames: tuple[int, ...]
    box: tuple[float, float, float, float]
    view: str = "front"


def overlap(a, b):
    area = lambda x: max(0.0, x[2] - x[0]) * max(0.0, x[3] - x[1])
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    return inter / max(1e-9, min(area(a), area(b)))


def compatible(label, target):
    """Conservative lexical attributes; missing attributes do not imply a match."""
    colors = {"red", "blue", "green", "yellow", "orange", "purple", "white"}
    a, b = (
        set(label.lower().replace("_", " ").split()),
        set(target.lower().replace("_", " ").split()),
    )
    if b & colors:
        return bool(a & b & colors) and not bool((a & colors) - b)
    return bool((a & b) - {"the", "a", "object", "target", "demonstrated"})


class EvidenceIndex:
    def __init__(self, capacity=256):
        self.capacity = capacity
        self.frames = OrderedDict()

    def add(self, frame, regions):
        if self.frames and frame < next(reversed(self.frames)):
            raise ValueError("Evidence must arrive in causal order")
        if any(r.frame != frame for r in regions):
            raise ValueError("Region/frame mismatch")
        self.frames[frame] = tuple(regions)
        while len(self.frames) > self.capacity:
            self.frames.popitem(last=False)


def key_features(
    index, keys, current, target, evidence, now, spatial_threshold=0.018, decay=16.0
):
    """Return [key, R/U/O/V]. Temporal keys also return per-source features.

    Identity and motion create separate contributions. P is in the query vector
    only. No future, discarded, or imagined visual feature is reconstructed.
    """
    if any(r.frame > now for r in current):
        raise ValueError("Future current observation")
    result, sources = [], []
    for key in keys:
        per_frame = []
        for frame in key.frames:
            if frame > now:
                raise ValueError("Future memory key")
            positive, irrelevance = [], 0.0
            for r in index.frames.get(frame, ()):
                if r.view != key.view:
                    continue
                alpha = overlap(r.box, key.box)
                if not alpha:
                    continue
                if r.distractor:
                    irrelevance = max(irrelevance, alpha * r.confidence)
                    continue
                if not compatible(r.label, target):
                    continue
                candidates = [
                    c
                    for c in current
                    if c.view == r.view
                    and not c.distractor
                    and compatible(c.label, r.label)
                ]
                obs = max((c.confidence for c in candidates), default=0.0)
                ctr = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                match = any(
                    np.linalg.norm(ctr(r.box) - ctr(c.box)) < spatial_threshold
                    for c in candidates
                )
                validity = (
                    -1.0 if obs > 0 and not match else math.exp(-(now - frame) / decay)
                )
                for fact in "ISM":
                    if fact in evidence and r.confidence > 0:
                        positive.append(
                            (
                                alpha,
                                r.confidence,
                                obs if fact == "S" else 0.0,
                                validity if fact == "S" else 1.0,
                            )
                        )
            den = max(1.0, sum(x[0] for x in positive))
            rv = [sum(x[0] * x[j] for x in positive) / den for j in (1, 2, 3)]
            per_frame.append([rv[0], irrelevance, rv[1], rv[2]])
        sources.append(np.asarray(per_frame, np.float32).reshape(-1, 4))
        result.append(np.mean(per_frame, axis=0) if per_frame else np.zeros(4))
    return np.asarray(result, np.float32).reshape(-1, 4), sources


def grid_keys(frames, grid, views=("front",), budget=512):
    keys = [
        MemoryKey((int(f),), (x / grid, y / grid, (x + 1) / grid, (y + 1) / grid), view)
        for f in frames
        for view in views
        for y in range(grid)
        for x in range(grid)
    ]
    if len(keys) > budget:
        raise ValueError("Native selection exceeded budget")
    return keys + [MemoryKey((), (0, 0, 1, 1))] * (budget - len(keys))
