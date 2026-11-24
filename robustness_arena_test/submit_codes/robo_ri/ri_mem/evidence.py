from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    frame: int
    label: str
    box: tuple[float, float, float, float]
    confidence: float
    distractor: bool = False
    motion: bool = False

    def __post_init__(self):
        x0, y0, x1, y1 = self.box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 and 0 <= self.confidence <= 1):
            raise ValueError("Invalid normalized region")


@dataclass(frozen=True)
class MemoryKey:
    frame: int
    box: tuple[float, float, float, float]
    historical: bool = True


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(1e-9, min(area_a, area_b))


class EvidenceIndex:
    def __init__(self, capacity=256, spatial_threshold=0.018, decay=16.0):
        self.capacity, self.spatial_threshold, self.decay = capacity, spatial_threshold, decay
        self.frames = OrderedDict()

    def add(self, frame: int, regions: list[Region]):
        if self.frames and frame < next(reversed(self.frames)):
            raise ValueError("Evidence must arrive causally")
        if any(r.frame != frame for r in regions):
            raise ValueError("Source frame mismatch")
        self.frames[frame] = list(regions)
        while sum(map(len, self.frames.values())) > self.capacity or len(self.frames) > self.capacity:
            self.frames.popitem(last=False)

    def features(self, keys: list[MemoryKey], current: list[Region], target: str, codes: str, now: int):
        out = np.zeros((len(keys), 4), dtype=np.float32)
        for j, key in enumerate(keys):
            if not key.historical:
                continue
            if key.frame > now:
                raise ValueError("Future memory key")
            facts = []
            for r in self.frames.get(key.frame, []):
                alpha = overlap(key.box, r.box)
                if r.distractor:
                    out[j, 1] = max(out[j, 1], alpha * r.confidence)
                    continue
                if r.label != target:
                    continue
                compatible = [c for c in current if c.label == r.label and not c.distractor]
                observed = max((c.confidence for c in compatible), default=0.0)
                center = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                matched = any(np.linalg.norm(center(r.box) - center(c.box)) < self.spatial_threshold for c in compatible)
                validity = -1.0 if observed > 0 and not matched else np.exp(-(now - r.frame) / self.decay)
                for code in codes:
                    if code == "I" or (code == "M" and r.motion):
                        facts.append((alpha, [r.confidence, 0, 1]))
                    elif code == "S":
                        facts.append((alpha, [r.confidence, observed, validity]))
            denom = max(1.0, sum(a for a, _ in facts))
            if facts:
                out[j, [0, 2, 3]] = sum((a * np.asarray(f) for a, f in facts)) / denom
        return out


def patch_keys(frames: list[int], grid: int, current_frame: int | None = None):
    return [MemoryKey(frame, (x / grid, y / grid, (x + 1) / grid, (y + 1) / grid), frame != current_frame)
            for frame in frames for y in range(grid) for x in range(grid)]
