"""Offline expert-action calibration. Supplement §2.3, Eqs.14–15/Table 3.

Inputs are decision points, NOT episode success labels. Each manifest row points
to a native preprocessed training batch plus query/key features. Clean/rendered
perturbed pairs must have identical robot-state and expert-action hashes.
"""

import json
from pathlib import Path
import numpy as np
from .rimem.head import initial_parameters, readout, save_head


def validate_manifest(path, expected_samples):
    path = Path(path)
    rows = json.loads(path.read_text())
    if len(rows) != expected_samples:
        raise ValueError(
            f"Paper calibration requires {expected_samples} decision points"
        )
    pairs = {}
    for row in rows:
        if (
            row.get("split") != "train"
            or row.get("query_source") != "expert_subtask"
            or row.get("augmentation") != "environment_render"
        ):
            raise ValueError(
                "Use official train trajectories, expert queries and environment-rendered augmentation"
            )
        for key in (
            "robot_state_sha256",
            "actions_sha256",
            "pair_id",
            "variant",
            "batch",
        ):
            if not row.get(key):
                raise ValueError(f"Missing calibration provenance: {key}")
        pairs.setdefault(row["pair_id"], []).append(row)
        row["batch"] = str((path.parent / row["batch"]).resolve())
        from .calibration_data import digest

        with np.load(row["batch"], allow_pickle=False) as data:
            if (
                digest(data["robot_state_raw"]) != row["robot_state_sha256"]
                or digest(data["actions_raw"]) != row["actions_sha256"]
            ):
                raise ValueError(
                    "Calibration batch does not match its robot/action provenance hashes"
                )
    for pair in pairs.values():
        if {r["variant"] for r in pair} != {"clean", "perturbed"} or len(pair) != 2:
            raise ValueError(
                "Every selected state needs exactly one clean and one perturbed observation"
            )
        if len({(r["robot_state_sha256"], r["actions_sha256"]) for r in pair}) != 1:
            raise ValueError(
                "Clean/perturbed calibration changes robot state or expert action"
            )
    return rows


def train_mme(checkpoint, manifest, output, backbone, seed=7):
    import jax
    import jax.numpy as jnp
    import optax
    from mme_vla_suite.policies import policy_config
    from mme_vla_suite.training import config
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from openpi.shared import nnx_utils
    from .rimem.mme import install

    install()
    records = validate_manifest(manifest, 32)
    policy = policy_config.create_trained_policy(
        config.get_config("mme_vla_suite"), Path(checkpoint), seed=seed
    )
    frozen_loss = nnx_utils.module_jit(policy._model.compute_loss)
    head = tuple(jnp.asarray(x) for x in initial_parameters())
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(3e-3, b1=0.9, b2=0.999, eps=1e-8)
    )
    opt_state = optimizer.init(head)
    rng = jax.random.key(seed)
    losses = []
    for step in range(32):
        row = records[step % len(records)]
        with np.load(row["batch"], allow_pickle=False) as data:
            batch = {k: jnp.asarray(data[k]) for k in data.files}
        # Native observations are already normalized/transformed; no repeated
        # normalization of expert actions or stored image/state embeddings.
        obs = {}
        for key, value in batch.items():
            if key.startswith("observation/"):
                node = obs
                parts = key.split("/")[1:]
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = value
        observation = HistAugObservation.from_dict(obs)
        rng, key = jax.random.split(rng)

        def loss_fn(parameters):
            bias, gate = readout(*parameters, batch["query"], batch["features"], xp=jnp)
            loss, _ = frozen_loss(
                key,
                observation,
                batch["actions"],
                train=False,
                ri_bias=bias[None],
                ri_gate=gate[None],
            )
            return jnp.mean(loss)

        loss, gradient = jax.value_and_grad(loss_fn)(head)
        updates, opt_state = optimizer.update(gradient, opt_state, head)
        head = optax.apply_updates(head, updates)
        losses.append(float(loss))
        if not np.isfinite(losses[-1]):
            raise FloatingPointError("Non-finite native flow-matching loss")
    save_head(
        output,
        *jax.device_get(head),
        dict(
            backbone=backbone,
            calibrated=True,
            seed=seed,
            optimizer="Adam",
            updates=32,
            decision_points=32,
            train_split=True,
            manifest=str(Path(manifest).resolve()),
            losses=losses,
        ),
    )


def torch_readout(W, beta, query, features):
    import torch

    y = query @ W + beta
    w = torch.nn.functional.softplus(y[:4])
    p = features @ (w * W.new_tensor([1, -1, -1, 1]))
    bias = 0.6 * torch.tanh(p / 0.6)
    uncertain = query[4] < 0.6
    bias = torch.where(uncertain, -0.6 * torch.tanh(w[1] * features[:, 1] / 0.6), bias)
    gate = torch.where(
        uncertain | ~torch.any(features != 0),
        W.new_tensor(1.0),
        1 + 0.08 * torch.tanh(y[4]),
    )
    return bias, gate


def train_simplememvla(checkpoint, manifest, output, seed=7):
    import torch
    from robomme_sim.policy import get_vla
    from .rimem.simplememvla import AttentionBridge

    records = validate_manifest(manifest, 8)
    torch.manual_seed(seed)
    model, _, _ = get_vla(str(checkpoint), "bfloat16", attn_implementation="sdpa")
    model.eval().requires_grad_(False)
    bridge = AttentionBridge(model)
    device = next(model.parameters()).device
    W, beta = [
        torch.nn.Parameter(torch.tensor(x, device=device, dtype=torch.float32))
        for x in initial_parameters()
    ]
    optimizer = torch.optim.AdamW(
        [W, beta], lr=3e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    losses = []
    try:
        for step in range(16):
            row = records[step % len(records)]
            with np.load(row["batch"], allow_pickle=False) as d:
                batch = {
                    k: torch.from_numpy(np.array(d[k])).to(device) for k in d.files
                }
            batch.pop("robot_state_raw")
            batch.pop("actions_raw")
            optimizer.zero_grad(set_to_none=True)
            # Teacher-forced expert subtask labels and native normalized actions.
            # Temporal-pair features are averaged by the exporter BEFORE this head.
            bias, gate = torch_readout(
                W, beta, batch.pop("query"), batch.pop("features")
            )
            bridge.set(bias, gate, batch.pop("video_positions"))
            # Native forward samples its own configured flow time and noise.
            result = model(**batch)
            loss = result.action_loss
            if loss is None or not loss.requires_grad:
                raise RuntimeError(
                    "Native action loss is not connected to RI-Mem attention"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W, beta], 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if not np.isfinite(losses[-1]):
                raise FloatingPointError("Non-finite native flow-matching loss")
    finally:
        bridge.close()
    save_head(
        output,
        W.detach().cpu().numpy(),
        beta.detach().cpu().numpy(),
        dict(
            backbone="simplememvla",
            calibrated=True,
            seed=seed,
            optimizer="AdamW",
            updates=16,
            decision_points=8,
            train_split=True,
            manifest=str(Path(manifest).resolve()),
            losses=losses,
            prefix_path="native teacher-forced forward; use native cached-prefix exporter for paper throughput",
        ),
    )
