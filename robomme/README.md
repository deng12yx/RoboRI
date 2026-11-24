# RoboMME-RI and RI-Mem

RoboMME perturbations follow the main paper, Sections 3 and 5 and Table 2. RI-Mem follows the supplement, Sections 1 and 2.3. This module is maintained on `robomme-submit`, separately from the Arena branch.

The scope is the **26 model baselines in Table 2**, clean evaluation and ten perturbations, plus RI-Mem on **FS+Mod, TD+Mod and SimpleMemVLA**. SmolVLA and memory-free OpenVLA-OFT, mentioned in Section 5.1, remain available as additional adapters. Human performance is not a policy baseline. Ablation and history-intervention experiments are excluded.

## Structure

| Path | Responsibility |
|---|---|
| `robomme_ri/protocol.py` | Sixteen tasks, official test episodes and ten event definitions |
| `scene.py`, `perturbations.py` | Simulator operations, control-step timing and admissibility |
| `baselines.py`, `policies.py` | Method registry and native MME/SimpleMemVLA adapters |
| `official.py`, `ponderpounce.py` | Service clients and native PonderPounce callbacks |
| `rimem/` | Shared SCQ, evidence index, bounded head and attention integrations |
| `calibrate.py`, `calibration_data.py` | Backbone-specific 35-parameter calibration |
| `evaluate.py` | Closed-loop execution, event logs and macro task success |
| `configs/robomme.json` | All 29 main method versions, sixteen tasks and fifty episodes |
| `notebooks/` | Three methods and four historical success-to-failure pairs |

## Setup and evaluation

```bash
cd robomme
python -m pip install -e '.[notebook,clients,video]'
export LAPSE_ROOT=/path/to/upstream/workspace
export MODEL_ROOT=/path/to/mme_vla_suite
export QWEN_MODEL=/path/to/Qwen3-VL-4B-Instruct
python -m robomme_ri.cli inventory
python -m robomme_ri.cli plan configs/robomme.json --output outputs/plan
python scripts/run_experiment.py configs/robomme.json --output outputs/experiment --gpus 0,1
```

`LAPSE_ROOT` contains the upstream benchmark and model repositories; it differs from this clone directory. Configure each policy's Python environment, checkpoint, LoRA and service endpoint. Native repositories, weights and training trajectories are external dependencies.

Each method runs in its own process. Optional `server_commands` is a list of command-argument lists; `{gpu}` selects the assigned device. The runner starts these services, waits for their ports and terminates its own processes on completion or interruption. Existing services retain their current GPU allocation.

```bash
python -m robomme_ri.cli evaluate configs/robomme.json \
  --method framesamp-modul --output outputs/fs
python -m robomme_ri.cli summarize outputs/fs/shard_0/rollouts.jsonl \
  --output outputs/fs/summary.json
```

`--shard k/n` partitions task/episode pairs, retaining paired conditions together. Completed episodes resume under the same configuration hash. Technical errors remain retryable. Summaries report event coverage separately from task success.

## Perturbations

| Group | Condition | RoboMME configuration |
|---|---|---|
| Remember | Onset Arrival | Native initialization; 0.65 m horizontal offset, at most 0.20 m elevation; settled by native step 20 |
| Remember | Mid-task Arrival | A later-stage object arrives after manipulation begins; 0.65 m offset, at most 0.25 m elevation; at most 12 steps |
| Remember | Slight Shift | Step 40 or later; 0.02 m horizontal displacement; retain the new endpoint |
| Remember | Remove & Return | Step 40; offset (0,0.65,0.25) m; restore within 12 steps and before use |
| Ignore | TV On | Steps 16–96; four colored blocks updated every eight steps |
| Ignore | Object Pass-by | Steps 32–96; a 0.68 m background trajectory |
| Ignore | Unrelated Object | Non-colliding object inserted at step 16 or removed at step 48 |
| Ignore | Target-like Object | A 5 cm green cube with a white cross; the same insertion/removal schedule |
| Ignore | Camera Shake | Step 16; horizontal sinusoid, 0.018 m amplitude coefficient; restore after three steps |
| Ignore | Light Change | Step 16; ambient illumination (2,2,2) or (0,0,0); restore after three steps |

Insertion/removal and bright/dark subtypes are balanced by episode parity. Event seeds depend on task, official environment seed, condition and subtype, independently of the method.

`EventHook` intercepts `BaseEnv.step`, including internal waypoint-planner steps. Initialization and online execution use separate clocks. Events add no waits or budget extensions. Joint-angle, end-effector and waypoint policies all use this evaluator.

Task-object events require a visible, reachable endpoint before use. The selector avoids grasped objects and hidden-answer disclosure. Invalid windows are recorded and restored. **The current conservative selector excludes task-object events on StopCube** to preserve its intrinsic timing; further task-specific work is required for full sixteen-task admissibility. Configuration coverage does not establish complete valid-rollout coverage.

## Baselines

All methods use the same ten-condition evaluator; no baseline-specific event implementation bypasses the control-step hook.

| Family | Table 2 variants | Count |
|---|---|---|
| Perceptual | FS/TD × Ctx/Mod/Exp; OFT FS/TD+Ctx; FS+Mod Retrieval; SimpleMemVLA; SAM2Act+; HAMLET | 12 |
| Symbolic | SimpleSG/GroundSG × Qwen/Oracle; OFT SimpleSG/GroundSG+Qwen | 6 |
| Perceptual-cognitive | MemoryVLA; MemER; PonderPounce | 3 |
| Recurrent | TTT+Ctx; TTT+Exp | 2 |
| Memory-free | pi0.5; SAM2Act; Diffusion Policy | 3 |

The configuration evaluates these 26 baselines and three RI-Mem variants under clean and all ten perturbations. SmolVLA and memory-free OpenVLA-OFT can be enabled by adding their registered identifiers to `methods`.

MME and SimpleMemVLA directly invoke native policies. Qwen variants load their released LoRA adapters; Oracle variants require online environment subgoals under separate identifiers. OFT/SmolVLA use HTTP, SAM2Act/DP/MemoryVLA use their deployed WebSocket wrappers, and HAMLET uses ZMQ. SAM2Act retains forty waypoints and its native control budget; MemoryVLA retains end-effector conversion and eight-step execution.

PonderPounce invokes the official `PonderPounceRoboMMEServer` callbacks with **one observation and action per control step**, retaining its cognition schedule and initial wait. Paired conditions share session identity and noise seeds while resetting episode state. Install [the official repository](https://github.com/worv-ai/PonderPounce) at `LAPSE_ROOT/PonderPounce`; the adapter was checked against revision `3d0ae369568e353a100a688c1f77e9d29c7f06d0`. Callback conformance is tested; pretrained performance remains unverified here.

To separate baseline Qwen inference from MME, run `scripts/serve_subgoals.py --native symbolic-simple-subgoal --adapter ADAPTER --qwen-model MODEL --port 8891` in its VLM environment and set `subgoal_endpoint: ws://127.0.0.1:8891`. This preserves native conversation and MemER keyframe behavior.

## Unified RI-Mem

Three backbones share structured task decomposition, asynchronous progress checks, RGB evidence indexing, retained-key alignment and a 35-parameter affine head.

- Query: `[S,I,P,M,c,1]`. Check completion every four decisions, at most sixteen requests per episode; confidence at least 0.85 advances one stage. Stage confidence is 1 for twenty decisions and 0.5 thereafter.
- Evidence: relevance, explicit irrelevance, observability and temporal validity follow supplement Equations 2–7. Stale spatial evidence receives validity −1; identity and motion remain position invariant.
- Readout: `bias = 0.6*tanh(priority/0.6)`; `gate = 1 + 0.08*tanh(y5)`. Low-confidence stages retain explicit irrelevance penalties only; empty evidence produces a unit gate.
- FS+Mod / TD+Mod preserve native 512-key selection. Bias enters memory-attention logits; the gate scales the output.
- SimpleMemVLA modifies historical video keys in full-attention layers 27 and 31 with bias and `log(gate)`. Other keys, masks, KV caches and action generation retain native behavior. Temporally merged keys average bounded source-frame priors at inference.

RGB evidence uses HSV regions and an explicit white-cross rule. Detector coverage needs further task-image validation. Policy-side evidence reads no simulator answers, event roles or future observations.

## Calibration

Each backbone has a separate calibrated head; VLA and auxiliary VLM remain frozen. MME uses 32 decision points, 32 Adam updates and horizon 20. SimpleMemVLA uses eight decision points, sixteen AdamW updates and horizon 30. Both use batch size 1, learning rate 0.003, zero weight decay and gradient clipping at 1.

```bash
python -m robomme_ri.cli calibrate configs/robomme.json \
  --backbone framesamp-modul --manifest calibration/fs/manifest.json \
  --output checkpoints/framesamp-modul.npz
```

`CalibrationWriter` exports paired NPZ batches and a manifest. Each pair contains clean and environment-rendered perturbed observations of identical raw robot states and expert actions. Loading recomputes both hashes and requires train-split, expert-subtask provenance.

MME batches include `query:[6]`, `features:[512,4]`, normalized expert `actions` and flattened native observation fields under `observation/`. SimpleMemVLA uses query/features, `video_positions` and native teacher-forced inputs: images/video, labels, normalized actions, state and masks. Temporally merged evidence features are averaged before the training head.

SimpleMemVLA currently uses a full teacher-forced forward; cached-prefix training throughput remains unverified. Learned heads are not bundled. Initialization-only heads are rejected for enhanced evaluation.

## Notebook and verification

The [notebook](notebooks/robomme_ri_demo.ipynb) references four local paired videos, instructions, keyframes and source records with SHA256 hashes. These historical examples differ from the paper protocol in timing, displacement and budgets. The shift example is 44.8 mm, versus the paper default of 20 mm. Six Ignore-category videos remain outside this module.

Open the notebook in Jupyter after installing the notebook dependencies. Outputs are cleared in Git; local MP4 and JPG links remain available without rerunning model evaluation.

## Reproduction status

The shared event engine, native memory-attention hooks, head/evidence calculations and PonderPounce callback cadence were checked locally. Full pretrained multi-baseline evaluation and calibrated RI-Mem results remain pending checkpoint and expert-data access. Native service endpoints must match the payloads in `official.py`; the model services and upstream training frameworks are external dependencies. The current StopCube exclusion and limited RGB-detector coverage require further validation before claiming complete paper reproduction.
