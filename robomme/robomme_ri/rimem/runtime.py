"""One shared SCQ/evidence/head implementation for all three backbones."""

import numpy as np
from .evidence import EvidenceIndex, key_features
from .head import load_head, numpy_readout
from .perception import HSVRegions
from .query import QueryTracker, Plan, Subgoal


class RIMem:
    def __init__(self, backbone, head_path, scq, ablation="full", detector=None):
        self.backbone, self.scq, self.ablation = backbone, scq, ablation
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

    def reset(self, instruction, seed=7):
        self.close()
        self.reset_index()
        if self.ablation == "reweight_only":
            plan = Plan(
                instruction,
                {"task": instruction},
                (Subgoal(instruction, "task", "task completed", frozenset("ISM")),),
            )
            self.tracker = QueryTracker(
                plan, lambda *a: {"complete": False, "confidence": 0}
            )
        else:
            self.tracker = QueryTracker(self.scq.parse(instruction), self.scq.verify)
        self.random_weights = (
            np.random.default_rng(seed).uniform(0.1, 2.5, 4).astype(np.float32)
        )

    def observe(self, front):
        self.now += 1
        self.current = self.detector(front, self.now)
        self.index.add(self.now, self.current)
        self.latest_image = np.asarray(front)

    def read(self, keys):
        query = (
            np.ones(6, np.float32)
            if self.ablation == "reweight_only"
            else self.tracker.update(self.latest_image)
        )
        subgoal = self.tracker.active
        target = self.tracker.plan.entities[subgoal.target]
        codes = subgoal.evidence
        if self.ablation == "reweight_only":
            # Query-free control: all referenced entities and all fact types.
            target = " ".join(self.tracker.plan.entities.values())
            codes = frozenset("ISM")
            query = np.array([1, 1, 1, 1, 1, 1], np.float32)
        feats, sources = key_features(
            self.index, keys, self.current, target, codes, self.now
        )
        bias, gate = numpy_readout(
            query,
            feats,
            self.parameters,
            ablation=self.ablation,
            random_weights=self.random_weights,
        )
        # Supplement §2.3: inference averages bounded SOURCE priors, not facts.
        for i, per_source in enumerate(sources):
            if len(per_source) > 1:
                bs, _ = numpy_readout(
                    query,
                    per_source,
                    self.parameters,
                    ablation=self.ablation,
                    random_weights=self.random_weights,
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
