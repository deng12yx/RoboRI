"""Exercise the native JAX memory-attention patch without loading policy checkpoints."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robo_ri.paths import load_paths, activate_fs
from robo_ri.ri_mem.bindings import install_fs_mod_readout

paths = load_paths()
activate_fs(paths)
from mme_vla_suite.models.integration import history_gemma
import jax
import jax.numpy as jnp
import numpy as np

attention = history_gemma.MemoryAttention()
x = jnp.ones((1, 2, 1024))
memory = jax.random.normal(jax.random.key(1), (1, 3, 1024))
mask = jnp.ones((1, 3), dtype=bool)
variables = attention.init(jax.random.key(2), x, memory, mask)
original = attention.apply(variables, x, memory, mask)
install_fs_mod_readout()
attention = history_gemma.MemoryAttention()
patched = attention.apply(variables, x, memory, mask, jnp.zeros((1, 3)), jnp.ones((1,)))
np.testing.assert_allclose(original, patched, atol=1e-5, rtol=1e-5)

def objective(bias):
    return jnp.square(attention.apply(variables, x, memory, mask, bias, jnp.ones((1,)))).mean()

gradient = jax.grad(objective)(jnp.zeros((1, 3)))
assert np.any(np.asarray(gradient) != 0)
from mme_vla_suite.models.integration.history_observation import HistAugObservation
assert "ri_key_bias" in HistAugObservation.__dataclass_fields__
from openpi.models.gemma import Config
config = Config(width=1024, depth=1, mlp_dim=64, num_heads=4, num_kv_heads=1, head_dim=256)
module = history_gemma.Module(configs=(config, config), embed_dtype="bfloat16", integration_type="modulation")
embedded = [jnp.ones((1, 1, 1024), jnp.bfloat16), jnp.ones((1, 2, 1024), jnp.bfloat16)]
positions = jnp.arange(3)[None]
attention_mask = jnp.ones((1, 3, 3), bool)
kwargs = dict(mem_seq=[None, memory.astype(jnp.bfloat16)], mem_mask=[None, mask],
              ri_key_bias=jnp.zeros((1, 3)), ri_gate=jnp.ones((1,)))
from flax import linen as nn
params = nn.Module.init(module, jax.random.key(3), embedded, positions, attention_mask, **kwargs)
out, _ = module.apply(params, embedded, positions, attention_mask, **kwargs)
assert out[-1].dtype == jnp.bfloat16 and out[-1].shape == (1, 2, 1024)
print("Native FS+Mod: zero-prior equivalence, differentiable bias, and observation fields passed")
print("Native scanned Gemma: RI arguments and bfloat16 output passed")
