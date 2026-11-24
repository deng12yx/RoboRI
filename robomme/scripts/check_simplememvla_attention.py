"""Check the native Transformers dispatch used by SimpleMemVLA, without weights."""

from pathlib import Path
from types import SimpleNamespace as NS
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from robomme_ri.rimem.simplememvla import AttentionBridge


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.config = NS(_attn_implementation="sdpa")
        self.num_key_value_groups = 2
        self.is_causal = True
        self.scaling = 0.5


model = NS(
    backbone=NS(
        model=NS(
            language_model=NS(
                layers=[NS(self_attn=Attention().eval()) for _ in range(32)]
            )
        )
    )
)
bridge = AttentionBridge(model)
m = bridge.targets[0]
torch.manual_seed(7)
q, k, v = torch.randn(1, 2, 3, 4), torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4)
mask = torch.zeros(1, 1, 3, 5)
mask[:, :, :, -1] = float("-inf")
native = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
modified = ALL_ATTENTION_FUNCTIONS.get_interface("robomme_ri_readout", None)
baseline = native(m, q, k, v, mask, scaling=0.5)[0]
bridge.set(torch.zeros(2), torch.tensor(1.0), torch.tensor([0, 1]))
zero = modified(m, q, k, v, mask, scaling=0.5)[0]
torch.testing.assert_close(zero, baseline)
bias = torch.tensor([0.4, -0.4], requires_grad=True)
bridge.set(bias, torch.tensor(0.98), torch.tensor([0, 1]))
changed = modified(m, q, k, v, mask, scaling=0.5)[0]
assert not torch.allclose(changed, baseline)
changed.sum().backward()
assert bias.grad is not None and bias.grad.norm() > 0
assert (
    model.backbone.model.language_model.layers[26].self_attn.config._attn_implementation
    == "sdpa"
)
bridge.clear()
torch.testing.assert_close(modified(m, q, k, v, mask, scaling=0.5)[0], baseline)
bridge.close()
assert m.config._attn_implementation == "sdpa"
print("SimpleMemVLA attention: identity, bias, gradient, layer scope and reset passed")
