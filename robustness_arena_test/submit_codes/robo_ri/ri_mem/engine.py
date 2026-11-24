from __future__ import annotations

from .evidence import EvidenceIndex
from .head import load_parameters, numpy_priors
from .query import QueryTracker
from .tracking import RegionTracker


class RIMem:
    def __init__(self, service, checkpoint, capacity=256):
        self.service, self.parameters = service, load_parameters(checkpoint)
        self.capacity = capacity
        self.tracker = None

    def reset(self, instruction, image):
        self.close()
        plan = self.service.plan(instruction)
        self.tracker = QueryTracker(plan, self.service.verify, image, instruction)
        self.index = EvidenceIndex(self.capacity)
        self.visual_tracker = RegionTracker()
        self.current, self.frame, self.decision = [], -1, 0

    def observe(self, image, frame):
        self.frame = frame
        if frame == 0:
            self.current = self.service.detect(image, self.tracker.plan.entities, frame)
            self.visual_tracker.seed(image, self.current)
        else:
            self.current = self.visual_tracker.update(image, frame)
        self.index.add(frame, self.current)

    def begin_decision(self, image):
        if self.decision > 0 and self.decision % 4 == 0:
            self.current = self.service.detect(image, self.tracker.plan.entities, self.frame)
            self.visual_tracker.seed(image, self.current)
            self.index.add(self.frame, self.current)
        self.tracker.update(self.decision, image)
        self.decision += 1

    def features(self, keys):
        active = self.tracker.active
        return self.index.features(keys, self.current, self.tracker.plan.entities[active["t"]], active["e"], self.frame)

    def priors(self, keys):
        return numpy_priors(self.parameters, self.tracker.vector(max(0, self.decision - 1)), self.features(keys))

    def close(self):
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
