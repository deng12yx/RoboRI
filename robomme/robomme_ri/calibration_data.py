"""Write paired decision batches with verifiable training provenance."""

import hashlib
import json
from pathlib import Path
import numpy as np


def digest(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


class CalibrationWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def add_pair(
        self,
        pair_id,
        robot_state,
        expert_actions,
        clean_batch,
        perturbed_batch,
        *,
        split,
        query_source,
        augmentation,
    ):
        if (split, query_source, augmentation) != (
            "train",
            "expert_subtask",
            "environment_render",
        ):
            raise ValueError(
                "Calibration requires environment-rendered official expert training data"
            )
        for variant, batch in (("clean", clean_batch), ("perturbed", perturbed_batch)):
            name = f"{len(self.rows):03d}_{variant}.npz"
            np.savez_compressed(
                self.directory / name,
                **batch,
                robot_state_raw=np.asarray(robot_state),
                actions_raw=np.asarray(expert_actions),
            )
            self.rows.append(
                dict(
                    pair_id=str(pair_id),
                    variant=variant,
                    batch=name,
                    split=split,
                    query_source=query_source,
                    augmentation=augmentation,
                    robot_state_sha256=digest(robot_state),
                    actions_sha256=digest(expert_actions),
                )
            )
        (self.directory / "manifest.json").write_text(
            json.dumps(self.rows, indent=2) + "\n"
        )
