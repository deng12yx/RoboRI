"""One shared SCQ/evidence/head implementation for all three backbones."""

import numpy as np
from .evidence import EvidenceIndex, key_features
from .head import load_head, numpy_readout
from .perception import HSVRegions
from .query import QueryTracker


class RIMem:
    def __init__(self, backbone, head_path, scq, detector=None):
        self.backbone, self.scq = backbone, scq
        self.parameters, self.metadata = load_head(head_path, backbone)
        self.detector = detector or HSVRegions()
        self.tracker = None
        self.capacity = 16384 if backbone == "simplememvla" else 256
        self.reset_index()

    def reset_index(self):
        self.index = EvidenceIndex(self.capacity)
        self.current = []
        self.now = -1
        self.trace = []

    def reset(self, instruction):
        self.close()
        self.reset_index()
        self.tracker = QueryTracker(self.scq.parse(instruction), self.scq.verify)

    def observe(self, front):
        self.now += 1
        self.current = self.detector(front, self.now)
        self.index.add(self.now, self.current)
        self.latest_image = np.asarray(front)

    def read(self, keys):
        query = self.tracker.update(self.latest_image)
        subgoal = self.tracker.active
        target = self.tracker.plan.entities[subgoal.target]
        codes = subgoal.evidence
        feats, sources = key_features(
            self.index, keys, self.current, target, codes, self.now
        )
        bias, gate = numpy_readout(
            query,
            feats,
            self.parameters,
        )
        # Supplement §2.3: inference averages bounded SOURCE priors, not facts.
        for i, per_source in enumerate(sources):
            if len(per_source) > 1:
                bs, _ = numpy_readout(
                    query,
                    per_source,
                    self.parameters,
                )
                bias[i] = bs.mean()
        self.trace.append(
            dict(
                frame=self.now,
                stage=self.tracker.stage,
                query=query.tolist(),
                target=target,
                keys=len(keys),
                evidence_keys=int(np.any(feats != 0, axis=1).sum()),
                bias_min=float(bias.min()) if len(bias) else 0.0,
                bias_max=float(bias.max()) if len(bias) else 0.0,
                gate=float(gate),
            )
        )
        return bias.astype(np.float32), np.float32(gate)

    def close(self):
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
