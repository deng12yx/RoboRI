from __future__ import annotations

import numpy as np

from .evidence import Region


class RegionTracker:
    def __init__(self):
        self.templates = []

    def seed(self, image, regions):
        self.templates = []
        h, w = image.shape[:2]
        for region in regions:
            x0, y0, x1, y1 = region.box
            crop = image[int(y0 * h):max(int(y0 * h) + 1, int(y1 * h)),
                         int(x0 * w):max(int(x0 * w) + 1, int(x1 * w))].copy()
            if crop.size:
                self.templates.append((region, crop))

    def update(self, image, frame):
        import cv2
        h, w = image.shape[:2]
        regions = []
        updated = []
        for previous, crop in self.templates:
            if crop.shape[0] > h or crop.shape[1] > w:
                continue
            correlation = cv2.matchTemplate(image, crop, cv2.TM_SQDIFF_NORMED)
            score, _, point, _ = cv2.minMaxLoc(correlation)
            if score > .30:
                updated.append((previous, crop))
                continue
            x, y = point
            box = (x / w, y / h, (x + crop.shape[1]) / w, (y + crop.shape[0]) / h)
            moved = np.linalg.norm(np.asarray(box[:2]) - np.asarray(previous.box[:2])) > .002
            region = Region(frame, previous.label, box, previous.confidence * max(0, 1 - score),
                            previous.distractor, moved)
            regions.append(region)
            updated.append((region, crop))
        self.templates = updated
        return regions
