"""SimpleMemVLA: native full-attention layers 27/31, video keys only.

Uses Transformers' attention dispatch, preserves KV caches and causal masks, and
delegates untouched layers to native SDPA. No key/value is scaled or replaced.
"""

import copy
import numpy as np
from .evidence import MemoryKey


def video_keys(processed, buffer):
    gh, gw = buffer._prompt_cache["grid_hw"]
    merge = int(buffer.vla.backbone.config.vision_config.spatial_merge_size)
    h, w = gh // merge, gw // merge
    keys = []
    for view, fa, fb in processed["video_patch_keys"]:
        for y in range(h):
            for x in range(w):
                keys.append(
                    MemoryKey(
                        (int(fa), int(fb)),
                        (x / w, y / h, (x + 1) / w, (y + 1) / h),
                        view,
                    )
                )
    positions = np.flatnonzero(
        processed["input_ids"][0].detach().cpu().numpy()
        == int(buffer.vla.backbone.config.video_token_id)
    )
    if len(keys) != len(positions):
        raise RuntimeError("Native temporal/spatial video-token mapping changed")
    return keys, positions


class AttentionBridge:
    def __init__(self, model):
        import torch
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.torch = torch
        self.layers = model.backbone.model.language_model.layers
        self.targets = []
        self.original = []
        sdpa = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
        if sdpa is None:
            raise RuntimeError("Transformers SDPA attention interface is unavailable")

        def attention(module, query, key, value, attention_mask, **kwargs):
            prior = getattr(module, "_ri_prior", None)
            if prior is not None:
                bias, gate, positions = prior
                n = key.shape[-2]
                full = torch.zeros(
                    (1, 1, 1, n), dtype=torch.float32, device=query.device
                )
                valid = positions < n
                # Joint attention gate: log(g) only on historical visual keys.
                full[..., positions[valid]] = bias[valid] + torch.log(gate)
                if attention_mask is None:
                    qn = query.shape[-2]
                    qi = torch.arange(n - qn, n, device=query.device)[:, None]
                    ki = torch.arange(n, device=query.device)[None, :]
                    attention_mask = torch.zeros(
                        (qn, n), dtype=torch.float32, device=query.device
                    ).masked_fill(ki > qi, float("-inf"))
                if attention_mask.dtype == torch.bool:
                    attention_mask = torch.zeros_like(
                        attention_mask, dtype=torch.float32
                    ).masked_fill(~attention_mask, float("-inf"))
                attention_mask = attention_mask.to(torch.float32) + full
                kwargs["is_causal"] = False  # explicit original causal mask above
            return sdpa(module, query, key, value, attention_mask, **kwargs)

        ALL_ATTENTION_FUNCTIONS.register("robomme_ri_readout", attention)
        for i in (27, 31):
            layer = self.layers[i]
            if not hasattr(layer, "self_attn"):
                raise RuntimeError(f"SimpleMemVLA layer {i} is not full attention")
            m = layer.self_attn
            self.original.append((m, m.config))
            m.config = copy.copy(m.config)
            m.config._attn_implementation = "robomme_ri_readout"
            self.targets.append(m)

    def set(self, bias, gate, positions):
        dev = next(self.targets[0].parameters()).device
        bias = self.torch.as_tensor(bias, dtype=self.torch.float32, device=dev)
        gate = self.torch.as_tensor(gate, dtype=self.torch.float32, device=dev)
        positions = self.torch.as_tensor(positions, dtype=self.torch.long, device=dev)
        for m in self.targets:
            m._ri_prior = (bias, gate, positions)

    def clear(self):
        for m in self.targets:
            m._ri_prior = None

    def close(self):
        self.clear()
        for m, cfg in self.original:
            m.config = cfg
