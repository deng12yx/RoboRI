"""Check native Qwen3.5 full-attention integration and frozen-backbone calibration."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
from robo_ri.ri_mem.bindings import SimpleMemBinding
from robo_ri.calibration import calibrate_simplemem

torch.set_num_threads(1)
torch.manual_seed(7)
config = Qwen3_5TextConfig(hidden_size=32, intermediate_size=64, num_attention_heads=2,
    num_key_value_heads=1, head_dim=8, num_hidden_layers=32, layer_types=["full_attention"] * 32)
config._attn_implementation = "sdpa"

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 32)
        self.layers = nn.ModuleList([Qwen3_5Attention(config, layer_idx=i) for i in (27, 31)])

    def model(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        rope = (torch.ones((1, x.shape[1], 8)), torch.zeros((1, x.shape[1], 8)))
        for layer in self.layers:
            x = x + layer(x, position_embeddings=rope, attention_mask=None)[0]
        return SimpleNamespace(last_hidden_state=x)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.action_head = nn.Linear(32, 7)

    def forward(self, input_ids, attention_mask, actions):
        hidden, _ = self._backbone_forward(input_ids, attention_mask)
        noise = torch.randn_like(actions)
        target = noise - actions
        return SimpleNamespace(action_loss=(self.action_head(hidden) - target).square().mean())

model = Policy().eval()
inputs = torch.tensor([[1, 2, 3]])
baseline = model.backbone.model(inputs).last_hidden_state.detach()
binding = SimpleMemBinding(SimpleNamespace(vla=model, _process=lambda: None), None)
binding.bias, binding.gate = torch.zeros(1, 3), torch.ones(1)
binding.historical = torch.tensor([[True, True, False]])
patched = model.backbone.model(inputs).last_hidden_state.detach()
torch.testing.assert_close(baseline, patched, atol=1e-6, rtol=1e-6)
binding.close()

sample = {"input__input_ids": inputs.numpy(), "input__attention_mask": np.ones((1, 3), np.int64),
    "actions": np.zeros((1, 3, 7), np.float32), "query": np.array([[1, 1, 0, 0, 1, 1]], np.float32),
    "features": np.array([[[1, 0, .8, -1], [0, 1, 0, 0], [0, 0, 0, 0]]], np.float32),
    "historical": np.array([[True, True, False]])}
before = {k: v.detach().clone() for k, v in model.state_dict().items()}
with tempfile.TemporaryDirectory() as directory:
    result = calibrate_simplemem(model, [sample], Path(directory) / "head.npz", steps=2)
    assert result["trainable_parameters"] == 35 and len(result["losses"]) == 2
for key, value in model.state_dict().items():
    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
print("Native SimpleMemVLA attention: identity and 35-parameter calibration passed; backbone weights unchanged")
