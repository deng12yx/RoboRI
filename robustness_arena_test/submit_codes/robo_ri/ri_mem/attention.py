from __future__ import annotations


def torch_attention(query, key, value, bias, gate, historical=None, mask=None, dedicated=False, causal=False, scale=None):
    import torch
    from torch.nn.functional import scaled_dot_product_attention
    length, keys = query.shape[-2], key.shape[-2]
    additions = bias[..., None, None, :].expand(*bias.shape[:-1], 1, length, keys)
    if not dedicated:
        if historical is None:
            raise ValueError("Joint attention requires a historical-key mask")
        additions = additions + gate.log()[..., None, None, None] * historical[..., None, None, :]
    if mask is not None:
        additions = additions + (torch.where(mask, 0.0, float("-inf")) if mask.dtype == torch.bool else mask)
    if causal:
        q_index = torch.arange(length, device=query.device) + keys - length
        k_index = torch.arange(keys, device=query.device)
        additions = additions.masked_fill(k_index[None, :] > q_index[:, None], float("-inf"))
    if query.shape[-3] != key.shape[-3]:
        repeat = query.shape[-3] // key.shape[-3]
        key, value = key.repeat_interleave(repeat, -3), value.repeat_interleave(repeat, -3)
    out = scaled_dot_product_attention(query, key, value, attn_mask=additions, dropout_p=0.0, scale=scale)
    return out * gate[..., None, None, None] if dedicated else out


def jax_memory_attention(query, key, value, mask, bias, gate):
    import jax
    import jax.numpy as jnp
    logits = jnp.einsum("bthd,bshd->bhts", query, key) * query.shape[-1] ** -0.5
    logits = logits + bias[:, None, None, :]
    logits = jnp.where(mask[:, None, None, :], logits, -jnp.inf)
    out = jnp.einsum("bhts,bshd->bthd", jax.nn.softmax(logits, -1), value)
    return out * gate[:, None, None, None]
