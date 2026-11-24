import json

import numpy as np
import pytest

from robo_ri.calibration import save_pair, calibration_samples, unpack_inputs


def test_calibration_pair_roundtrip_rejects_tampering(tmp_path):
    batch = {"inputs": {"image": {"front": np.zeros((1, 2, 2, 3), np.float32)},
                        "state": np.zeros((1, 8), np.float32)},
             "actions": np.zeros((1, 20, 7), np.float32),
             "query": np.array([[1, 0, 0, 0, 1, 1]], np.float32),
             "features": np.zeros((1, 2, 4), np.float32)}
    rows = save_pair(tmp_path, "pair", batch, batch, np.zeros(10), "pick butter", "tv_on")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"split": "train", "samples": rows}))
    samples = calibration_samples(manifest)
    assert len(samples) == 2
    assert unpack_inputs(samples[0])["image"]["front"].shape == (1, 2, 2, 3)
    (tmp_path / rows[0]["file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        calibration_samples(manifest)


def test_calibration_refuses_test_split_and_unpaired_actions(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"split": "test", "samples": []}))
    with pytest.raises(ValueError, match="training"):
        calibration_samples(manifest)
    with pytest.raises(ValueError, match="identical expert actions"):
        save_pair(tmp_path, "bad", {"actions": np.zeros(1)}, {"actions": np.ones(1)},
                  np.zeros(10), "pick butter", "tv_on")
