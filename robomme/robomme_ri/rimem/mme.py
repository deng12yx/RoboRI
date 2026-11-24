"""JAX MME integration at native memory attention, without checkpoint edits.

Patch only the method bodies in memory. Dynamic bias/gate are explicit arguments
of sample_actions/compute_loss, so JIT never captures stale per-episode Python
state. Parameter names/shapes, RoPE, original mask and memory selection persist.
"""

import inspect
import textwrap


def _replace(function, replacements, globals_extra=None):
    function = inspect.unwrap(function)
    source = textwrap.dedent(inspect.getsource(function))
    source = source[source.index("def ") :]  # unwrap decorators, reapply below
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(
                f"Unsupported MME source layout at {function.__name__}: {old!r}"
            )
        source = source.replace(old, new)
    scope = dict(function.__globals__)
    scope.update(globals_extra or {})
    exec(compile(source, "<robomme-ri/mme-readout>", "exec"), scope)
    return scope[function.__name__]


_installed = False


def install():
    global _installed
    if _installed:
        return
    import flax.linen as nn
    from flax.linen.module import wrap_method_once
    from mme_vla_suite.models.integration import (
        history_gemma as gemma,
        history_pi0 as pi0,
    )

    attention = _replace(
        gemma.MemoryAttention.__call__,
        [
            (
                "B, mem_len, mem_width = mem_seq.shape",
                """ri_bias = ri_gate = None
    if isinstance(mem_mask, tuple):
        mem_mask, ri_bias, ri_gate = mem_mask
    B, mem_len, mem_width = mem_seq.shape""",
            ),
            (
                "masked_logits = jnp.where(attn_mask, logits, -2.3819763e38)",
                """if ri_bias is not None:
        logits = logits + ri_bias[:, None, None, None, :]
    masked_logits = jnp.where(attn_mask, logits, -2.3819763e38)""",
            ),
            (
                'return out_einsum("BTNH,NHD->BTD", encoded)',
                """result = out_einsum("BTNH,NHD->BTD", encoded)
    return result if ri_gate is None else result * ri_gate.reshape(-1, 1, 1)""",
            ),
        ],
    )
    gemma.MemoryAttention.__call__ = wrap_method_once(nn.compact(attention))
    # The modulation-only mask payload carries the native mask plus two arrays.
    module_call = _replace(
        gemma.Module.__call__,
        [
            (
                'mem_mask: Sequence[at.Bool[at.Array, "b lmem"] | None] | None = None,',
                "mem_mask: Sequence[object] | None = None,",
            ),
        ],
    )
    gemma.Module.__call__ = wrap_method_once(module_call)
    for name, signature in (
        ("sample_actions", 'num_steps: int | at.Int[at.Array, ""] = 10,'),
        ("compute_loss", "train: bool = False,"),
    ):
        old = getattr(pi0.HistoryPi0, name)
        updated = _replace(
            old,
            [
                (signature, signature + "\n    ri_bias=None, ri_gate=None,"),
                (
                    "mem_mask=[None, mem_mask],",
                    "mem_mask=[None, (mem_mask, ri_bias, ri_gate)],",
                ),
            ],
        )
        setattr(pi0.HistoryPi0, name, updated)
    _installed = True


def attach(policy, rimem, representation):
    """Call install() BEFORE constructing the policy; attach AFTER reset()."""
    import jax.numpy as jnp
    from .evidence import MemoryKey, grid_keys

    mem = policy.mem_buffer
    original_prepare = policy._prepare_history
    native_frame = mem.get_frame_sampling_indices
    native_token = mem.prepare_token_dropping
    selected = []

    def frames(step, budget, tokens):
        indices = native_frame(step, budget, tokens)
        selected[:] = grid_keys(indices, int(tokens**0.5), budget=budget)
        return indices

    def tokens(step, budget, gather, *args, **kwargs):
        indices = mem.filter_token_dropping_indices(
            mem.get_token_dropping_indices(), step, budget
        )
        selected[:] = [
            MemoryKey(
                (int(t),), (p % 8 / 8, p // 8 / 8, (p % 8 + 1) / 8, (p // 8 + 1) / 8)
            )
            for t, v, p in indices
        ]
        selected.extend([MemoryKey((), (0, 0, 1, 1))] * (budget - len(selected)))
        return native_token(step, budget, gather, *args, kept_indices=indices, **kwargs)

    mem.get_frame_sampling_indices = frames
    mem.prepare_token_dropping = tokens

    def prepare(inputs):
        result = original_prepare(inputs)
        bias, gate = rimem.read(selected)
        policy._sample_kwargs.update(
            ri_bias=jnp.asarray(bias[None]), ri_gate=jnp.asarray([gate])
        )
        return result

    policy._prepare_history = prepare

    def detach():
        mem.get_frame_sampling_indices = native_frame
        mem.prepare_token_dropping = native_token
        policy._prepare_history = original_prepare
        policy._sample_kwargs.pop("ri_bias", None)
        policy._sample_kwargs.pop("ri_gate", None)

    return detach
