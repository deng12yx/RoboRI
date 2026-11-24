import numpy as np
import pytest

from robo_ri.ri_mem.evidence import EvidenceIndex, MemoryKey, Region
from robo_ri.ri_mem.head import initial_parameters, numpy_priors, torch_head
from robo_ri.ri_mem.query import Plan, InvalidPlan, QueryTracker
from robo_ri.retrieval import select_segment


def test_moved_object_invalidates_spatial_history_but_not_identity():
    index = EvidenceIndex()
    index.add(0, [Region(0, "butter", (.1, .1, .2, .2), 1)])
    current = [Region(10, "butter", (.5, .5, .6, .6), .9)]
    key = [MemoryKey(0, (0, 0, 1, 1))]
    spatial = index.features(key, current, "butter", "S", 10)
    identity = index.features(key, current, "butter", "I", 10)
    assert spatial[0, 3] == -1 and spatial[0, 2] == pytest.approx(.9)
    assert identity[0, 3] == 1 and identity[0, 2] == 0
    assert numpy_priors(initial_parameters(), [1, 0, 0, 0, 1, 1], spatial)[0][0] < 0


def test_irrelevance_is_independent_of_current_subgoal():
    index = EvidenceIndex()
    index.add(0, [Region(0, "other task object", (.1, .1, .2, .2), 1),
                  Region(0, "TV", (.5, .5, .8, .8), 1, distractor=True)])
    keys = [MemoryKey(0, (.1, .1, .2, .2)), MemoryKey(0, (.5, .5, .8, .8))]
    features = index.features(keys, [], "butter", "S", 10)
    assert np.array_equal(features[0], [0, 0, 0, 0])
    assert np.array_equal(features[1], [0, 1, 0, 0])
    bias, gate = numpy_priors(initial_parameters(), [1, 0, 0, 0, .5, 1], features)
    assert bias[0] == 0 and bias[1] < 0 and gate == 1


def test_current_keys_and_future_keys():
    index = EvidenceIndex()
    index.add(0, [Region(0, "cube", (.1, .1, .2, .2), 1)])
    assert not index.features([MemoryKey(0, (0, 0, 1, 1), False)], [], "cube", "I", 0).any()
    with pytest.raises(ValueError, match="Future"):
        index.features([MemoryKey(2, (0, 0, 1, 1))], [], "cube", "I", 1)


def test_head_bounds_uncertainty_empty_evidence_and_gradients():
    import torch
    head = torch_head()
    assert sum(p.numel() for p in head.parameters()) == 35
    query = torch.tensor([[1., 1, 1, 1, 1, 1], [1., 0, 0, 0, .5, 1]])
    features = torch.tensor([[[.9, 0, .8, -1], [0, .8, 0, 0]], [[.8, 0, .1, 1], [0, 1, 0, 0]]])
    bias, gate = head(query, features)
    assert torch.all(bias.abs() <= .6) and torch.all((gate >= .92) & (gate <= 1.08))
    assert bias[1, 0] == 0 and gate[1] == 1
    for row in range(2):
        expected_b, expected_g = numpy_priors(initial_parameters(), query[row].numpy(), features[row].numpy())
        np.testing.assert_allclose(bias[row].detach(), expected_b, atol=1e-7)
        np.testing.assert_allclose(gate[row].detach(), expected_g, atol=1e-7)
    (bias.sum() + gate.sum()).backward()
    assert head.W.grad.abs().sum() > 0
    assert head(query, torch.zeros_like(features))[1].tolist() == [1, 1]


def test_attention_preserves_mask_and_zero_prior_identity():
    import torch
    from robo_ri.ri_mem.attention import torch_attention
    torch.manual_seed(4)
    q, k, v = torch.randn(1, 2, 3, 8), torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)
    mask = torch.tensor([[[[True, True, False, False]]]])
    baseline = torch.nn.functional.scaled_dot_product_attention(q, k.repeat_interleave(2, 1),
                                                               v.repeat_interleave(2, 1), attn_mask=mask)
    actual = torch_attention(q, k, v, torch.zeros(1, 4), torch.ones(1),
                             torch.ones(1, 4, dtype=torch.bool), mask)
    torch.testing.assert_close(actual, baseline)
    changed_v = v.clone()
    changed_v[:, :, 2:] = 1e6
    changed = torch_attention(q, k, changed_v, torch.zeros(1, 4), torch.ones(1),
                              torch.ones(1, 4, dtype=torch.bool), mask)
    torch.testing.assert_close(changed, baseline)


def test_query_validation_and_confidence_are_deterministic():
    raw = {"e": {"r": "red cube"}, "s": [{"d": "pick up cube", "t": "r", "c": "cube held", "e": "IS"}]}
    tracker = QueryTracker(Plan.parse(raw), lambda *args: {"complete": False, "confidence": .9}, np.zeros((4, 4, 3)), "pick cube")
    assert tracker.vector(19).tolist() == [1, 1, 0, 0, 1, 1]
    assert tracker.vector(20)[4] == .5
    tracker.close()
    raw["s"].append(dict(raw["s"][0]))
    with pytest.raises(InvalidPlan):
        Plan.parse(raw)


def test_retrieval_is_segment_based():
    embeddings = np.array([[1., 0]] * 4 + [[0., 1]] * 4, dtype=np.float16)
    assert select_segment(embeddings, [0, 1]) == [4, 5, 6, 7]
    assert select_segment(embeddings[:2], [0, 1]) == [0, 1]


def test_repetition_count_does_not_leak_across_actions():
    raw = {"e": {"r": "sauce"}, "s": [
        {"d": "pick sauce", "t": "r", "c": "sauce held", "e": "IS"},
        {"d": "pour sauce once", "t": "r", "c": "one pour completed", "e": "PM"},
        {"d": "pour sauce again", "t": "r", "c": "two pours completed", "e": "PM"}]}
    Plan.parse(raw, "Pick up the sauce and pour it twice.")
    with pytest.raises(InvalidPlan, match="2 pour"):
        Plan.parse({"e": raw["e"], "s": raw["s"][:2]}, "Pick up the sauce and pour it twice.")
