import numpy as np
import pytest
from robomme_ri.rimem.evidence import (
    Region,
    MemoryKey,
    EvidenceIndex,
    key_features,
    grid_keys,
)
from robomme_ri.rimem.head import (
    initial_parameters,
    numpy_readout,
    save_head,
    load_head,
)
from robomme_ri.rimem.query import Plan, InvalidPlan, QueryTracker


def test_stale_position_is_penalized_but_identity_is_retained():
    index = EvidenceIndex()
    index.add(0, [Region(0, "red object", (0.1, 0.1, 0.2, 0.2), 0.9)])
    key = MemoryKey((0,), (0.1, 0.1, 0.2, 0.2))
    current = [Region(10, "red object", (0.6, 0.6, 0.7, 0.7), 0.95)]
    spatial, _ = key_features(index, [key], current, "red cube", {"S"}, 10)
    identity, _ = key_features(index, [key], current, "red cube", {"I"}, 10)
    np.testing.assert_allclose(spatial, [[0.9, 0, 0.95, -1]])
    np.testing.assert_allclose(identity, [[0.9, 0, 0, 1]])
    sb, _ = numpy_readout([1, 0, 0, 0, 1, 1], spatial)
    ib, _ = numpy_readout([0, 1, 0, 0, 1, 1], identity)
    assert sb[0] < ib[0]


def test_explicit_distractor_not_temporarily_irrelevant_target():
    index = EvidenceIndex()
    index.add(
        0,
        [
            Region(0, "green object", (0.1, 0.1, 0.2, 0.2), 1),
            Region(
                0, "white-cross distractor", (0.3, 0.3, 0.4, 0.4), 1, distractor=True
            ),
        ],
    )
    keys = [
        MemoryKey((0,), (0.1, 0.1, 0.2, 0.2)),
        MemoryKey((0,), (0.3, 0.3, 0.4, 0.4)),
    ]
    feats, _ = key_features(index, keys, [], "red cube", {"S"}, 1)
    assert feats[0, 1] == 0 and feats[1, 1] == 1


def test_index_and_padding_and_future_leakage():
    index = EvidenceIndex(2)
    for t in range(3):
        index.add(t, [])
    assert list(index.frames) == [1, 2]
    keys = grid_keys([1, 2], 4)
    assert len(keys) == 512 and sum(bool(k.frames) for k in keys) == 32
    with pytest.raises(ValueError, match="Future"):
        key_features(index, [MemoryKey((9,), (0, 0, 1, 1))], [], "red", {"S"}, 2)


def test_head_bounds_uncertain_gate_and_35_parameters(tmp_path):
    W, b = initial_parameters()
    assert W.size + b.size == 35
    f = np.array([[1, 0, 1, -1], [0, 1, 0, 0], [0, 0, 0, 0]], np.float32)
    bias, gate = numpy_readout([1, 1, 0, 0, 1, 1], f)
    assert np.all(abs(bias) < 0.6) and 0.92 < gate < 1.08
    uncertain, g = numpy_readout([1, 1, 0, 0, 0.5, 1], f)
    assert g == 1 and uncertain[0] == 0 and uncertain[1] < 0
    removed, g = numpy_readout([1, 1, 0, 0, 0.5, 1], f, ablation="no_irrelevance")
    np.testing.assert_array_equal(removed, np.zeros(3))
    assert numpy_readout([1, 1, 0, 0, 1, 1], np.zeros((3, 4)))[1] == 1
    path = tmp_path / "head.npz"
    save_head(path, W, b, {"backbone": "framesamp-modul", "calibrated": False})
    with pytest.raises(ValueError, match="Uncalibrated"):
        load_head(path, "framesamp-modul")
    with pytest.raises(ValueError, match="separate"):
        load_head(path, "simplememvla", True)


def test_query_compiler_and_stage_confidence():
    raw = {
        "e": {"r": "red cube"},
        "s": [
            {
                "d": "pick up red cube",
                "t": "r",
                "c": "first pickup completed",
                "e": "IS",
            }
        ],
    }
    plan = Plan.parse("pick up red cube", raw)
    with pytest.raises(InvalidPlan):
        Plan.parse("pick up red cube 3 times", raw)
    tracker = QueryTracker(plan, lambda *a: {"complete": False, "confidence": 1})
    try:
        values = [tracker.update(np.zeros((4, 4, 3), np.uint8))[4] for _ in range(70)]
        assert values[:20] == [1] * 20 and values[20:] == [0.5] * 50
        assert tracker.requests <= 16
    finally:
        tracker.close()
