from types import SimpleNamespace
import numpy as np
import pytest
from robomme_ri.history_controls import (
    Control,
    patch_permutation,
    attach_simple,
    simple_processed,
)
from robomme_ri.rimem.evidence import MemoryKey
from robomme_ri.calibration_data import CalibrationWriter
from robomme_ri.calibrate import validate_manifest


def test_shuffle_preserves_spatial_slots_and_current_observation():
    boxes = ((0, 0, 0.5, 0.5), (0.5, 0.5, 1, 1))
    keys = [MemoryKey((t,), box) for t in range(8) for box in boxes]
    permutation = patch_permutation(keys, 7, np.random.default_rng(3))
    assert not np.array_equal(permutation, np.arange(16))
    assert permutation[-2:].tolist() == [14, 15]
    for i, j in enumerate(permutation):
        assert keys[i].box == keys[j].box
    assert sorted(permutation.tolist()) == list(range(16))


def test_oracle_keeps_current_frame_and_uses_historical_masks():
    control = Control("oracle-mem")
    control.now = 2
    control.masks = [
        np.eye(4, dtype=bool),
        np.zeros((4, 4), bool),
        np.zeros((4, 4), bool),
    ]
    assert control.keep(MemoryKey((0,), (0, 0, 0.5, 0.5)))
    assert not control.keep(MemoryKey((1,), (0, 0, 1, 1)))
    assert control.keep(MemoryKey((2,), (0, 0, 1, 1)))
    assert not control.keep(MemoryKey((), (0, 0, 1, 1)))


def test_simple_no_history_repeats_exactly_current_frame_and_restores():
    native = lambda view: list(range(6))
    buffer = SimpleNamespace(
        _history_positions=native,
        _video_patch_keys=lambda: [],
        _preprocess_patch_pixels=lambda k: None,
        _buffers={"front": [0, 1, 2, 3, 4, 5]},
    )
    detach = attach_simple(buffer, Control("no-history"))
    assert buffer._history_positions("front") == [5, 5]
    detach()
    assert buffer._history_positions is native


def test_simple_shuffle_reorders_features_after_native_cache_preparation():
    control = Control("shuffled-history")
    control.rng = np.random.default_rng(3)
    keys = [("front", t, t + 1) for t in range(0, 20, 2)]
    cache = {k: np.full((1, 4), i) for i, k in enumerate(keys)}
    processed = {"video_patch_keys": keys, "video_patch_pixels": cache}
    changed = simple_processed(processed, control)
    assert (
        changed["video_patch_keys"] != keys
        and changed["video_patch_keys"][-1] == keys[-1]
    )
    assert processed["video_patch_keys"] == keys
    assert changed["video_patch_pixels"] is cache


def test_simple_oracle_masks_full_clip_and_individual_patch_consistently():
    images = [np.zeros((2, 2, 3), np.uint8) + i for i in range(3)]
    positions = lambda view: [0, 1, 2]
    buffer = SimpleNamespace(
        _history_positions=positions,
        _history_video=lambda view: np.stack(buffer._buffers[view]),
        _preprocess_patch_pixels=lambda key: np.stack(
            [
                buffer._buffers["observation.images.front"][key[1]],
                buffer._buffers["observation.images.front"][key[2]],
            ]
        ),
        _buffers={"observation.images.front": images.copy()},
        _frames_seen={"observation.images.front": 3},
        history_image_keys=["observation.images.front"],
    )
    control = Control("oracle-mem")
    control.now = 2
    control.masks = [np.zeros((2, 2), bool)] * 3
    detach = attach_simple(buffer, control)
    clip = buffer._history_video("observation.images.front")
    pair = buffer._preprocess_patch_pixels(("front", 1, 2))
    np.testing.assert_array_equal(clip[-2:], pair)
    assert (clip[:-1] == 128).all() and (clip[-1] == 2).all()
    np.testing.assert_array_equal(buffer._buffers["observation.images.front"], images)
    detach()


def test_calibration_provenance_is_checked_against_batch(tmp_path):
    writer = CalibrationWriter(tmp_path)
    writer.add_pair(
        "point-1",
        np.arange(8),
        np.ones((20, 8)),
        {},
        {},
        split="train",
        query_source="expert_subtask",
        augmentation="environment_render",
    )
    manifest = tmp_path / "manifest.json"
    assert len(validate_manifest(manifest, 2)) == 2
    path = tmp_path / writer.rows[0]["batch"]
    np.savez(path, robot_state_raw=np.arange(8) + 1, actions_raw=np.ones((20, 8)))
    with pytest.raises(ValueError, match="hashes"):
        validate_manifest(manifest, 2)
