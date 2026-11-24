from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .ri_mem.head import initial_parameters, torch_head, jax_priors


def unpack_inputs(sample):
    result = {}
    for key, value in sample.items():
        if not key.startswith("input__"):
            continue
        parts = key.removeprefix("input__").split("__")
        parent = result
        for part in parts[:-1]:
            parent = parent.setdefault(part, {})
        parent[parts[-1]] = value
    return result


def save_pair(directory, pair_id, clean, perturbed, robot_state, expert_subtask, condition):
    """Write native, normalized training batches from a matched environment replay.

    Each batch contains inputs, actions, query, features, and (for SimpleMemVLA) historical.
    The caller renders both observations at the same expert state before preprocessing them.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if condition == "clean":
        raise ValueError("A calibration pair needs a perturbation")
    if not np.array_equal(clean["actions"], perturbed["actions"]):
        raise ValueError("Matched states must retain identical expert actions")
    records = []
    for name, batch in (("clean", clean), (condition, perturbed)):
        arrays = {key: value for key, value in batch.items() if key != "inputs"}
        def flatten(tree, prefix):
            for key, value in tree.items():
                if isinstance(value, dict):
                    flatten(value, prefix + key + "__")
                elif value is not None:
                    arrays[prefix + key] = np.asarray(value)
        flatten(batch["inputs"], "input__")
        path = directory / f"{pair_id}_{name}.npz"
        np.savez_compressed(path, **arrays)
        records.append({"file": path.name, "pair_id": pair_id, "condition": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "robot_state_sha256": hashlib.sha256(np.asarray(robot_state).tobytes()).hexdigest(),
            "expert_subtask": expert_subtask, "rendering": "environment", "query_source": "expert_subtask"})
    return records


def calibration_samples(manifest):
    manifest = Path(manifest)
    data = json.loads(manifest.read_text())
    if data.get("split") != "train":
        raise ValueError("Calibration accepts official training examples only")
    samples, pairs = [], {}
    for row in data["samples"]:
        if row["rendering"] != "environment" or row["query_source"] != "expert_subtask":
            raise ValueError("Use rendered perturbations and expert-subtask calibration queries")
        path = (manifest.parent / row["file"]).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Calibration sample hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            sample = {k: arrays[k] for k in arrays.files}
        if sample["query"].shape[-1] != 6 or sample["features"].shape[-1] != 4:
            raise ValueError("Invalid RI-Mem features")
        if "historical" in sample and np.any(sample["features"][~sample["historical"].astype(bool)] != 0):
            raise ValueError("Current and nonvisual keys must have zero evidence features")
        previous = pairs.setdefault(row["pair_id"], {})
        if row["condition"] in previous:
            raise ValueError("Duplicate condition in calibration pair")
        previous[row["condition"]] = (row["robot_state_sha256"], hashlib.sha256(sample["actions"].tobytes()).hexdigest())
        samples.append(sample)
    if not samples:
        raise ValueError("Empty calibration set")
    for pair_id, pair in pairs.items():
        if "clean" not in pair or len(pair) < 2 or len(set(pair.values())) != 1:
            raise ValueError(f"Need paired clean/perturbed robot states and expert actions: {pair_id}")
    return samples


def calibrate_simplemem(vla, samples, output, steps=16, seed=7):
    import torch
    from .ri_mem.bindings import SimpleMemBinding, enable_simplemem_calibration
    torch.manual_seed(seed)
    enable_simplemem_calibration(vla)
    vla.eval()
    device = next(vla.parameters()).device
    head = torch_head().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, betas=(.9, .999), eps=1e-8, weight_decay=0)
    binding = SimpleMemBinding(SimpleNamespace(vla=vla, _process=lambda: None), None)
    losses = []
    try:
        for step in range(steps):
            sample = samples[step % len(samples)]
            query, features = (torch.as_tensor(sample[k], device=device, dtype=torch.float32) for k in ("query", "features"))
            binding.bias, binding.gate = head(query, features)
            binding.historical = torch.as_tensor(sample["historical"], device=device, dtype=torch.bool)
            inputs = {k: torch.as_tensor(v, device=device) for k, v in unpack_inputs(sample).items()}
            inputs["actions"] = torch.as_tensor(sample["actions"], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = vla(**inputs).action_loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Native action flow-matching loss is missing or nonfinite")
            loss.backward()
            if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()):
                raise RuntimeError("No action-loss gradient reaches the RI-Mem head")
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        if any(p.grad is not None for p in vla.parameters()):
            raise RuntimeError("Frozen VLA received parameter gradients")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        head.save(output)
    finally:
        binding.close()
    return {"steps": steps, "trainable_parameters": sum(p.numel() for p in head.parameters()), "losses": losses}


def calibrate_fs(model, samples, output, steps=32, seed=7):
    import jax
    import jax.numpy as jnp
    import optax
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    parameters = jax.tree.map(jnp.asarray, initial_parameters())
    optimizer = optax.chain(optax.clip_by_global_norm(1), optax.adam(.003, b1=.9, b2=.999, eps=1e-8))
    state = optimizer.init(parameters)
    losses = []

    def objective(theta, sample, rng):
        bias, gate = jax_priors(theta, sample["query"], sample["features"])
        inputs = unpack_inputs(sample)
        # Native normalized observations, memory tensors, and actions come from the official training loader.
        observation = HistAugObservation.from_dict(inputs).replace(ri_key_bias=bias, ri_gate=gate)
        losses, _ = model.compute_loss(rng, observation, sample["actions"], train=False)
        return jnp.mean(losses)

    value_grad = jax.value_and_grad(objective)
    for step in range(steps):
        sample = jax.tree.map(jnp.asarray, samples[step % len(samples)])
        loss, grads = value_grad(parameters, sample, jax.random.fold_in(jax.random.key(seed), step))
        if not np.isfinite(float(loss)) or not any(np.any(np.asarray(g) != 0) for g in jax.tree.leaves(grads)):
            raise RuntimeError("Native action loss did not produce finite RI-Mem gradients")
        updates, state = optimizer.update(grads, state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        losses.append(float(loss))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **jax.tree.map(np.asarray, parameters))
    return {"steps": steps, "trainable_parameters": 35, "losses": losses}
