"""Shared 35-parameter head: supplement Eqs. 8–13 and Table 3."""

import numpy as np


def initial_parameters():
    W = np.zeros((6, 5), np.float32)
    W[0, 2], W[1, 3], W[0, 4], W[1, 4] = 1.0, 0.2, -0.12, 0.08
    beta = np.r_[np.log(np.expm1([0.45, 2.4, 0.15, 0.2])), 0.0].astype(np.float32)
    return W, beta


def readout(W, beta, query, features, *, xp=np):
    """Map the six-dimensional query and N x 4 evidence to bounded priors."""
    y = query @ W + beta
    weights = xp.logaddexp(y[:4], 0.0)
    priorities = features @ (weights * xp.asarray([1.0, -1.0, -1.0, 1.0]))
    full_bias = 0.6 * xp.tanh(priorities / 0.6)
    fallback_bias = -0.6 * xp.tanh(weights[1] * features[:, 1] / 0.6)
    uncertain = query[4] < 0.6
    bias = xp.where(uncertain, fallback_bias, full_bias)
    gate = xp.where(uncertain | ~xp.any(features != 0), 1.0, 1.0 + 0.08 * xp.tanh(y[4]))
    return bias, gate


def numpy_readout(query, features, parameters=None):
    W, beta = initial_parameters() if parameters is None else parameters
    return readout(
        W,
        beta,
        np.asarray(query, np.float32),
        np.asarray(features, np.float32),
    )


def save_head(path, W, beta, metadata):
    import json

    if np.shape(W) != (6, 5) or np.shape(beta) != (5,):
        raise ValueError("RI-Mem head must have exactly 35 parameters")
    np.savez(
        path,
        W=np.asarray(W, np.float32),
        beta=np.asarray(beta, np.float32),
        metadata=json.dumps(metadata),
    )


def load_head(path, backbone, allow_uncalibrated=False):
    import json

    with np.load(path, allow_pickle=False) as d:
        W, beta, meta = d["W"], d["beta"], json.loads(str(d["metadata"]))
    if (
        W.shape != (6, 5)
        or beta.shape != (5,)
        or not np.isfinite(W).all()
        or not np.isfinite(beta).all()
    ):
        raise ValueError("Invalid RI-Mem checkpoint")
    if meta.get("backbone") != backbone:
        raise ValueError("Use a separate calibrated head for each backbone")
    if not meta.get("calibrated") and not allow_uncalibrated:
        raise ValueError("Uncalibrated head: train it before reporting RI-Mem results")
    return (W, beta), meta
