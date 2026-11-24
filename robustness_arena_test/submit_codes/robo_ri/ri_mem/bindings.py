from __future__ import annotations

import copy
import inspect
import types

import numpy as np

from .evidence import MemoryKey, patch_keys


class SimpleMemBinding:
    """Attach key priors to native full-attention layers without rebuilding the KV cache."""

    def __init__(self, policy, engine):
        import torch
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from .attention import torch_attention
        self.policy, self.engine, self.bias, self.historical, self.gate = policy, engine, None, None, None
        self.original_process = policy._process
        self.original_implementations = []

        def readout(module, query, key, value, attention_mask, scaling=None, **kwargs):
            from torch.nn.functional import scaled_dot_product_attention
            if self.bias is None or getattr(module, "layer_idx", -1) not in (27, 31):
                if query.shape[1] != key.shape[1]:
                    repeat = query.shape[1] // key.shape[1]
                    key, value = key.repeat_interleave(repeat, 1), value.repeat_interleave(repeat, 1)
                output = scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                    dropout_p=0.0, is_causal=attention_mask is None and query.shape[-2] > 1, scale=scaling)
            else:
                count = key.shape[-2]
                if count < self.bias.shape[-1]:
                    raise ValueError("Native KV length is shorter than the aligned prefix")
                extra = count - self.bias.shape[-1]
                bias = torch.nn.functional.pad(self.bias.to(query.device), (0, extra))
                historical = torch.nn.functional.pad(self.historical.to(query.device), (0, extra))
                output = torch_attention(query, key, value, bias, self.gate.to(query.device), historical,
                    attention_mask, causal=attention_mask is None, scale=scaling)
            return output.transpose(1, 2).contiguous(), None

        self.interface_name = f"ri_mem_{id(self)}"
        ALL_ATTENTION_FUNCTIONS[self.interface_name] = readout
        for module in policy.vla.backbone.modules():
            if getattr(module, "layer_idx", -1) in (27, 31) and hasattr(module, "q_proj"):
                previous = module.config
                module.config = copy.copy(previous)
                module.config._attn_implementation = self.interface_name
                self.original_implementations.append((module, previous))
        if len(self.original_implementations) != 2:
            self.close()
            raise RuntimeError("Expected native full-attention layers 27 and 31")

        def process(*args, **kwargs):
            result = self.original_process(*args, **kwargs)
            self.align(result)
            return result
        policy._process = process

    def align(self, inputs):
        import torch
        policy = self.policy
        token = policy.vla.backbone.config.video_token_id
        positions = (inputs["input_ids"][0] == token).nonzero().flatten()
        bias = np.zeros(inputs["input_ids"].shape[-1], dtype=np.float32)
        historical = np.zeros_like(bias, dtype=bool)
        cursor = 0
        gates = []
        merge = policy.vla.backbone.config.vision_config.spatial_merge_size
        for camera, grid in zip(policy.history_image_keys, inputs["video_grid_thw"].tolist(), strict=True):
            temporal, height, width = grid
            height, width = height // merge, width // merge
            local = policy._history_positions(camera)
            offset = policy._frames_seen[camera] - len(policy._buffers[camera])
            frames = [offset + i for i in local]
            if len(frames) != temporal * 2:
                raise ValueError("Unexpected temporal merge: expected two source frames per video patch")
            for t in range(temporal):
                sources = frames[2 * t:2 * t + 2]
                spatial = [(x / width, y / height, (x + 1) / width, (y + 1) / height)
                           for y in range(height) for x in range(width)]
                source_priors = []
                for frame in sources:
                    keys = [MemoryKey(frame, box, frame < self.engine.frame) for box in spatial]
                    priors, gate = self.engine.priors(keys)
                    source_priors.append(priors)
                    gates.append(gate)
                indices = positions[cursor:cursor + height * width].cpu().numpy()
                bias[indices] = np.mean(source_priors, axis=0)
                historical[indices] = any(f < self.engine.frame for f in sources)
                cursor += height * width
        if cursor != len(positions):
            raise ValueError("Video tokens do not match native temporal/spatial layout")
        self.bias = torch.from_numpy(bias[None])
        self.historical = torch.from_numpy(historical[None])
        self.gate = torch.tensor([next((g for g in gates if g != 1.0), 1.0)], dtype=torch.float32)

    def close(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.policy._process = self.original_process
        for module, config in self.original_implementations:
            module.config = config
        self.original_implementations.clear()
        if self.interface_name in ALL_ATTENTION_FUNCTIONS:
            del ALL_ATTENTION_FUNCTIONS[self.interface_name]


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Unsupported native source revision: {old[:70]!r}")
    return source.replace(old, new, 1)


def install_fs_mod_readout():
    """Extend loaded native modules in memory; upstream files and parameter names stay intact."""
    import re
    from mme_vla_suite.models.integration import history_observation as observation
    from mme_vla_suite.models.integration import history_gemma as gemma
    from mme_vla_suite.models.integration import history_pi0 as pi0
    if getattr(gemma, "_ri_mem_installed", False):
        return
    src = inspect.getsource(observation)
    src = _replace_once(src, 'class HistAugObservation(_Observation):',
        'class HistAugObservation(_Observation):\n    ri_key_bias: at.Float[at.Array, "b l1"] | None = None\n    ri_gate: at.Float[at.Array, "b"] | None = None')
    src = _replace_once(src, 'static_image_emb=data.get("static_image_emb", None),',
        'ri_key_bias=data.get("ri_key_bias"), ri_gate=data.get("ri_gate"),\n            static_image_emb=data.get("static_image_emb", None),')
    src = _replace_once(src, 'result = super().to_dict()',
        'result = super().to_dict()\n        result["ri_key_bias"] = self.ri_key_bias\n        result["ri_gate"] = self.ri_gate')
    src = _replace_once(src, 'base_obs: _Observation,',
        'base_obs: _Observation,\n        ri_key_bias=None, ri_gate=None,')
    src = _replace_once(src, 'images=base_obs.images,',
        'ri_key_bias=ri_key_bias, ri_gate=ri_gate,\n            images=base_obs.images,')
    src = _replace_once(src, 'static_image_emb=observation.static_image_emb,',
        'ri_key_bias=observation.ri_key_bias, ri_gate=observation.ri_gate,\n        static_image_emb=observation.static_image_emb,')
    exec(compile(src, observation.__file__, "exec"), observation.__dict__)

    src = inspect.getsource(gemma)
    src = _replace_once(src, 'def __call__(self, x, mem_seq, mem_mask):',
        'def __call__(self, x, mem_seq, mem_mask, ri_key_bias=None, ri_gate=None):')
    src = _replace_once(src, 'probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)',
        'if ri_key_bias is not None:\n            masked_logits = masked_logits + ri_key_bias[:, None, None, None, :]\n        probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)')
    src = _replace_once(src, 'return out_einsum("BTNH,NHD->BTD", encoded)',
        'out = out_einsum("BTNH,NHD->BTD", encoded)\n        return out if ri_gate is None else (out * ri_gate[:, None, None]).astype(out.dtype)')
    src = _replace_once(src, '        deterministic=True,\n    ):',
        '        deterministic=True,\n        ri_key_bias=None,\n        ri_gate=None,\n    ):')
    src = _replace_once(src, 'mem_attn(x, mem_seq[-1], mem_mask[-1])',
        'mem_attn(x, mem_seq[-1], mem_mask[-1], ri_key_bias, ri_gate)')
    anchor = '            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=mem_seq, 5=mem_mask, 6=deterministic'
    src = _replace_once(src, anchor, '                nn.broadcast,\n                nn.broadcast,\n' + anchor)
    src = _replace_once(src, '        deterministic: bool = True,',
        '        deterministic: bool = True,\n        ri_key_bias: at.Float[at.Array, "b lmem"] | None = None,\n        ri_gate: at.Float[at.Array, "b"] | None = None,')
    src = _replace_once(src, '            mem_mask,\n            deterministic,',
        '            mem_mask,\n            deterministic,\n            ri_key_bias,\n            ri_gate,')
    exec(compile(src, gemma.__file__, "exec"), gemma.__dict__)

    src = inspect.getsource(pi0)
    src, count = re.subn(r'(?m)^(\s*)mem_mask=\[None, mem_mask\],$',
        lambda m: m[0] + '\n' + m[1] + 'ri_key_bias=observation.ri_key_bias,\n' + m[1] + 'ri_gate=observation.ri_gate,', src)
    if count != 2:
        raise RuntimeError(f"Expected two native memory readouts, found {count}")
    exec(compile(src, pi0.__file__, "exec"), pi0.__dict__)
    gemma._ri_mem_installed = True


def bind_fs_policy(policy, engine):
    if policy.config.num_views != 1 or policy.config.token_per_image != 16:
        raise ValueError("This Arena FS+Mod binding expects one camera and 4×4 memory patches")
    original = policy._input_transform

    def transform(inputs):
        result = original(inputs)
        frames = policy.mem_buffer.get_frame_sampling_indices(policy.step_idx, policy.config.budget,
                                                              policy.config.token_per_image)
        keys = patch_keys(frames, 4, current_frame=policy.step_idx)
        valid_count = int(np.asarray(result["static_mask"]).sum())
        if len(keys) != valid_count:
            raise ValueError(f"Native FS key layout mismatch: {len(keys)} versus {valid_count}")
        priors, gate = engine.priors(keys)
        padded = np.zeros(policy.config.budget, dtype=np.float32)
        padded[np.asarray(result["static_mask"], bool)] = priors
        result["ri_key_bias"], result["ri_gate"] = padded, np.float32(gate)
        return result
    policy._input_transform = transform


def enable_simplemem_calibration(vla):
    """Keep the frozen backbone differentiable with respect to readout priors."""
    def forward(self, input_ids, attention_mask, position_ids=None, labels=None, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs["use_cache"] = False
        output = self.backbone.model(input_ids=input_ids, attention_mask=attention_mask,
                                     position_ids=position_ids, **kwargs)
        hidden = output.last_hidden_state
        return hidden, self._sparse_ce_loss(hidden, labels) if labels is not None else None
    vla.requires_grad_(False)
    vla._backbone_forward = types.MethodType(forward, vla)
