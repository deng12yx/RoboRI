from __future__ import annotations

import numpy as np


def initial_parameters():
    w = np.zeros((6, 5), dtype=np.float32)
    w[0, 2], w[1, 3], w[0, 4], w[1, 4] = 1, 0.2, -0.12, 0.08
    b = np.r_[np.log(np.expm1([0.45, 2.4, 0.15, 0.2])), 0].astype(np.float32)
    return {"W": w, "beta": b}


def numpy_priors(parameters, query, features):
    y = np.asarray(query) @ parameters["W"] + parameters["beta"]
    weights = np.logaddexp(0, y[:4])
    p = np.asarray(features) @ (weights * [1, -1, -1, 1])
    gate = 1 + 0.08 * np.tanh(y[4])
    if query[4] < 0.6:
        p = -weights[1] * features[:, 1]
        gate = 1.0
    if not np.any(features):
        gate = 1.0
    return 0.6 * np.tanh(p / 0.6), gate


def load_parameters(path):
    with np.load(path, allow_pickle=False) as data:
        parameters = {k: np.asarray(data[k], dtype=np.float32) for k in ("W", "beta")}
    if parameters["W"].shape != (6, 5) or parameters["beta"].shape != (5,):
        raise ValueError("Expected a 35-parameter RI-Mem head")
    if not all(np.isfinite(v).all() for v in parameters.values()):
        raise ValueError("Nonfinite RI-Mem checkpoint")
    return parameters


def torch_head(parameters=None):
    import torch
    from torch import nn

    class ReadoutHead(nn.Module):
        def __init__(self):
            super().__init__()
            p = initial_parameters() if parameters is None else parameters
            self.W = nn.Parameter(torch.tensor(p["W"]))
            self.beta = nn.Parameter(torch.tensor(p["beta"]))

        def forward(self, query, features):
            y = query @ self.W + self.beta
            weights = torch.nn.functional.softplus(y[..., :4])
            signed = weights * weights.new_tensor([1, -1, -1, 1])
            p = (features * signed.unsqueeze(-2)).sum(-1)
            confident = query[..., 4] >= 0.6
            p = torch.where(confident.unsqueeze(-1), p, -weights[..., 1, None] * features[..., 1])
            enabled = confident & features.ne(0).any(dim=(-2, -1))
            gate = torch.where(enabled, 1 + 0.08 * y[..., 4].tanh(), torch.ones_like(y[..., 4]))
            return 0.6 * (p / 0.6).tanh(), gate

        def save(self, path):
            np.savez(path, W=self.W.detach().cpu().numpy(), beta=self.beta.detach().cpu().numpy())

    return ReadoutHead()


def jax_priors(parameters, query, features):
    import jax
    import jax.numpy as jnp
    y = query @ parameters["W"] + parameters["beta"]
    weights = jax.nn.softplus(y[..., :4])
    p = (features * (weights * jnp.array([1, -1, -1, 1]))[..., None, :]).sum(-1)
    confident = query[..., 4] >= 0.6
    p = jnp.where(confident[..., None], p, -weights[..., 1, None] * features[..., 1])
    enabled = confident & jnp.any(features != 0, axis=(-2, -1))
    gate = jnp.where(enabled, 1 + 0.08 * jnp.tanh(y[..., 4]), 1.0)
    return 0.6 * jnp.tanh(p / 0.6), gate
