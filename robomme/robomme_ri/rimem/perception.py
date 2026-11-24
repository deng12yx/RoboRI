"""RGB-only HSV/connected-component evidence extraction (supplement §2.3)."""

import numpy as np
from .evidence import Region


class HSVRegions:
    def __init__(self, min_pixels=16, max_fraction=0.25):
        self.min_pixels, self.max_fraction = min_pixels, max_fraction

    def __call__(self, rgb, frame, view="front"):
        import cv2

        hsv = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2HSV)
        H, W = hsv.shape[:2]
        # OpenCV hue in [0,180). Low-saturation lighting/background is excluded.
        ranges = {
            "red": ((0, 10), (170, 180)),
            "orange": ((10, 22),),
            "yellow": ((22, 35),),
            "green": ((35, 85),),
            "blue": ((90, 130),),
            "purple": ((130, 170),),
        }
        result = []
        for color, spans in ranges.items():
            mask = np.zeros((H, W), np.uint8)
            for lo, hi in spans:
                mask |= (
                    (hsv[:, :, 0] >= lo)
                    & (hsv[:, :, 0] < hi)
                    & (hsv[:, :, 1] >= 90)
                    & (hsv[:, :, 2] >= 50)
                ).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if not self.min_pixels <= area <= self.max_fraction * H * W:
                    continue
                patch = hsv[y : y + h, x : x + w]
                white = (patch[:, :, 1] < 45) & (patch[:, :, 2] > 200)
                # A cross must have a white horizontal and vertical central arm.
                cy, cx = h // 2, w // 2
                cross = (
                    w >= 8
                    and h >= 8
                    and white[max(0, cy - 1) : cy + 2, :].mean() > 0.5
                    and white[:, max(0, cx - 1) : cx + 2].mean() > 0.5
                )
                distractor = bool(color == "green" and cross)
                result.append(
                    Region(
                        frame,
                        "white-cross distractor" if distractor else f"{color} object",
                        (x / W, y / H, (x + w) / W, (y + h) / H),
                        min(0.99, 0.5 + 0.5 * area / (w * h)),
                        view,
                        distractor,
                    )
                )
        return result
