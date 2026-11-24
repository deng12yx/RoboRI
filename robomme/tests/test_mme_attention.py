"""Run in the MME environment; no checkpoint or GPU is required."""

import numpy as np
import pytest


def test_native_attention_equivalence_dynamic_jit_and_gradient():
    jax = pytest.importorskip("jax")
    pytest.importorskip("mme_vla_suite")
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_gemma import MemoryAttention
    from robomme_ri.rimem.mme import install

    x = jax.random.normal(jax.random.key(2), (1, 2, 1024))
    memory = jax.random.normal(jax.random.key(3), (1, 4, 1024))
    mask = jnp.array([[True, True, True, False]])
    original = MemoryAttention()
    variables = original.init(jax.random.key(1), x, memory, mask)
    expected = jax.jit(lambda a, b: original.apply(variables, a, b, mask))(x, memory)
    install()
    modified = MemoryAttention()

    @jax.jit
    def apply(bias, gate):
        return modified.apply(variables, x, memory, (mask, bias, gate))

    actual = apply(jnp.zeros((1, 4)), jnp.ones(1))
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=1e-5)
    changed = apply(jnp.array([[0.5, -0.5, 0, 999]]), jnp.ones(1))
    assert not np.allclose(changed, expected)
    padded = apply(jnp.array([[0, 0, 0, 999]]), jnp.ones(1))
    np.testing.assert_allclose(padded, expected, atol=3e-6)
    gradient = jax.grad(lambda b: jnp.sum(apply(b, jnp.ones(1))))(jnp.zeros((1, 4)))
    assert (
        np.isfinite(gradient).all()
        and np.linalg.norm(gradient) > 0
        and gradient[0, -1] == 0
    )
