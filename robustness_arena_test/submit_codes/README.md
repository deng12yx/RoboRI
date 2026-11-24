# RoboMemArena-RI / RI-Mem

This submission contains only RoboMemArena. The protocol follows `../papers/Robo_RI_CVPR27.pdf` (main paper §§3–5) and `../papers/Robo_RI_CVPR27_supplement.pdf` (§§1–2, Tables 1–3). Clean is a separate control, not one of the ten perturbations. No `late`, `onset_delay`, `arrive_late`, or `takeback_late` conditions enter a run.

## Layout

```text
configs/                 task/seed matrices and local asset paths
robo_ri/perturbations.py  MuJoCo event implementation and validity records
robo_ri/baselines.py      native policy adapters and dual-system rollouts
robo_ri/ri_mem/           shared query, evidence index, head, attention bindings
robo_ri/collect.py        paired environment-rendered expert calibration samples
robo_ri/calibration.py    frozen-backbone action-loss calibration
robo_ri/runner.py         native scorer, episode records, resume
robo_ri/analysis.py       task-macro TSR, coverage, paired successful execution times
tools/                   native integration checks, notebook/video export
appendix_videos/          ten verified historical failure / clean-success pairs
validation/              checks and real checkpoint smoke results
robomemarena_ri_examples.ipynb
```

The existing repositories, model assets, and interpreters under `robustness_arena_test` are reused. They are not copied into this directory. `robo_ri/paths.py` derives paths from this checkout; `--paths configs/paths.local.json` overrides them. Paths inside that JSON are relative to its own directory unless absolute. No dataset or checkpoint download occurs automatically.

## Conditions

The clock includes ten settling steps: recorded step 10 is the first policy observation. Control is 20 Hz; wall-clock inference latency never advances the event clock.

| Canonical condition | Intervention |
|---|---|
| `onset_arrival` | First task object arrives from a 0.90 m horizontal offset during steps 10–50; must precede first manipulation. |
| `mid_arrival` | A later required object arrives from a 1.80 m offset after first manipulation starts, normally completed by step 120, earlier if necessary. |
| `mid_shift` | Eligible future object moves 0.025 or 0.040 m along a feasible horizontal direction after step 80. |
| `remove_return` | Eligible future object leaves and returns to its saved pose within steps 80–160, before manipulation; at most 4 s. |
| `tv_on` | Eight animated stripes on the background screen during steps 40–280. |
| `object_passby` | Non-colliding humanoid traverses a 2.90 m background path during steps 40–280. |
| `unrelated_object` | Non-colliding blue sphere, diameter 0.068 m; insertion or removal. |
| `target_like_object` | Category-colored 0.058 × 0.040 × 0.050 m box with a white cross; insertion or removal. |
| `camera_shake` | Main-camera local y/z offsets up to 0.010/0.008 m, steps 40–43 (0.15 s), then restored. |
| `light_change` | Ambient/diffuse ×1.9 or ×0.35, steps 40–43 (0.15 s), then restored. |

The sphere/box insertion window is 40–280; removal occurs at step 160. Subtypes are assigned deterministically across seeds, identically for every method. `handin` / `slide` are recorded for task-object events. Shift magnitudes are balanced separately from presentation style.

`TARGETS` supplies native BDDL names for all 26 tasks; task 13 follows the native cookies-then-butter scorer. The scheduler uses environment state for feasibility and timing, while policy inputs remain native camera observations, robot state, and instruction. It does not expose scorer stages, event labels, object poses, or success to RI-Mem. Native stage checks are guarded while a scripted event controls the scored object, preventing scripted motion from earning task credit.

An event must actually finish and be camera-visible. Missing stage opportunities, late object return, infeasible displacement, and non-visible events are recorded as invalid. `paper_default_timing_met` additionally separates mid-arrivals that could not finish by the nominal step 120. These are reported explicitly rather than silently converted into clean episodes. The implementation uses geometric proximity and native stage boundaries as manipulation-boundary proxies; it does not claim to recover an unobserved policy's intentions.

## Methods

| ID | Native execution / checkpoint |
|---|---|
| `fs_mod` | Released Arena all-26 FS+Mod checkpoint, 512 memory keys, 4×4 patches, 20-action chunks. |
| `fs_mod_retrieval` | Same FS+Mod, frozen SigLIP visual segment retrieval; causal, current-image query, threshold .923, minimum segment 4 frames. |
| `simplememvla` | Released Arena SimpleMemVLA video-history adapter; 10-action replanning. |
| `predimem` | Native asynchronous Qwen S2 + Arena pi0.5 S1 loop, recent frames + native keyframe bank. |
| `traceflow` | Released upper/lower TraceFlow services and native asynchronous planner loop; suite-specific heads/gates retained. |
| `memer` | Official MemER subgoal/keyframe predictor + Arena pi0.5 action policy. Requires an explicitly configured MemER adapter. |
| `pi05` | Released Arena pi0.5 action weights, task instruction directly, no historical memory. |
| `fs_mod_ri` | FS+Mod with shared RI-Mem and its own calibrated 35-parameter head. |
| `simplememvla_ri` | SimpleMemVLA with shared RI-Mem and its own calibrated 35-parameter head. |

There is no substitute random policy behind a baseline ID. MemER's local Arena training checkpoint is not supplied by the current harness; `memer_adapter` must identify the intended weights. No RoboMME checkpoint is silently selected. MemER's short-window and empty-keyframe boundary errors are handled without changing its prompt or keyframe mechanism. π0.5 uses the existing PrediMem action checkpoint, as in the local Arena baseline.

RI-Mem uses a frozen Qwen3-VL-4B-Instruct for instruction decomposition and asynchronous completion checks (every four decisions, one pending request, maximum 16 requests, confidence threshold .85). The plan compiler validates references, nonempty evidence codes, distinct repeated completion conditions, and explicit pickup/pour counts. Queries are `[S, I, P, M, confidence, 1]`; stage confidence becomes .5 after 20 decisions without advancement. Only explicit distractors receive an irrelevance penalty. Spatial evidence becomes invalid when a compatible currently visible object has moved; identity and motion facts remain valid. Query-conditioned softplus weights, ±.6 bounded key biases, the .92–1.08 gate, and the low-confidence irrelevance-only fallback follow the supplement.

`EvidenceStream` / `RegionTracker` ground sparse regions with the frozen VLM and propagate them with observation-only template tracking. This is an explicit perception implementation choice because the PDFs do not provide an Arena detector/checkpoint. Motion marks come from observed region motion, not simulator state. Source-frame/patch alignment uses retained native keys only. FS+Mod changes dedicated memory attention; SimpleMemVLA changes historical video-key readout in full-attention layers 27 and 31, including temporally merged source-frame priors and cache reuse. The native memory selector and stored visual features are preserved. Only the affine head is trained; no new VLA weights are added.

## Environment and a small run

Run commands from the LAPSE repository root. The existing separate environments are required: JAX FS+Mod, PyTorch SimpleMemVLA, and the Arena evaluator have different dependency versions.

```bash
export PYTHONPATH="$PWD/robustness_arena_test/submit_codes${PYTHONPATH:+:$PYTHONPATH}"
robustness_arena_test/.venv-eval/bin/python -m robo_ri --help
robustness_arena_test/.venv-eval/bin/python -m robo_ri plan \
  --config robustness_arena_test/submit_codes/configs/smoke.json \
  --out robustness_arena_test/submit_codes/runs/smoke
```

`smoke.json` contains two tasks × two seeds × eleven settings × four FS/Simple original/enhanced methods. Select a method to avoid loading unrelated policies. Set `--limit 1` for a single episode. `paper.json` contains 26 tasks × 51 seeds (50–100) × eleven settings × nine methods: 131,274 total episodes, including 119,340 perturbed episodes. It is a planned matrix, not a claim that this full matrix has been run.

Start the appropriate FS server in another terminal, then run its jobs:

```bash
robomme_policy_learning/.venv/bin/python -m robo_ri serve-fs \
  --method fs_mod --port 8801 --gpu 7
robustness_arena_test/.venv-eval/bin/python -m robo_ri run \
  --manifest robustness_arena_test/submit_codes/runs/smoke/manifest.json \
  --method fs_mod --gpu 4
```

For retrieval use `--method fs_mod_retrieval --port 8802 --paths ...`; configure `retrieval_checkpoint` with the existing SigLIP weights. For RI use `--method fs_mod_ri --port 8803 --paths ...` after calibration. `qwen_device` is relative to visible GPUs; e.g. `--gpu 7,4` with `qwen_device: "cuda:1"` separates JAX and the auxiliary VLM. Start one rollout worker per server initially; reset releases episode-specific buffers.

SimpleMemVLA runs in process:

```bash
robustness_arena_test/.venv-simplemem/bin/python -m robo_ri run \
  --manifest robustness_arena_test/submit_codes/runs/smoke/manifest.json \
  --method simplememvla --gpu 4
```

Its RI variant uses the same command with `--method simplememvla_ri --paths ...`. The tested native attention API is Transformers 5.13.1 / PyTorch 2.4.1; use the existing SimpleMemVLA environment rather than upgrading the Arena evaluator.

For π0.5, PrediMem, and MemER, start the existing S1 server, then run the corresponding manifest jobs in the evaluator (PrediMem) or the ms-swift environment (MemER, with the Arena simulation dependencies installed):

```bash
CUDA_VISIBLE_DEVICES=7 robustness_arena_test/.venv-vla/bin/python \
  robustness_arena_test/serve_vla.py --port 8800
```

TraceFlow needs the native suite-specific upper/lower services. This helper creates an in-memory-equivalent launcher snapshot and disables only the original environment-worker launch; it does not modify third-party source files or the service flags:

```bash
python robustness_arena_test/submit_codes/tools/serve_traceflow.py \
  --suite transferring --upper-gpu 4 --lower-gpu 7
robustness_arena_test/.venv-tf-predimem/bin/python -m robo_ri run \
  --manifest path/to/transferring_manifest.json --method traceflow --gpu 5
```

Use separate task-filtered manifests for sequence, transferring, counting, and occlusion because released TraceFlow heads differ by suite. `--prepare-only` generates and syntax-checks its launcher without loading models. The config files are ordinary JSON and can be reduced to a few tasks/seeds before planning.

## Calibration loop

No trained RI-Mem head is fabricated or bundled as a reproduced paper result. The default head initialization is exposed for verification; inference requires a saved calibrated checkpoint.

1. Copy `configs/calibration_replays.example.json` and point it to a small subset of **official training** HDF5 demonstrations. Each group supplies native flattened MuJoCo `states` and physical `actions [T,7]`. Supply expert stage text, evidence codes, entity descriptions, stage start indices and selected decision indices. The example paths/stage indices are schema examples, not downloadable training data.
2. `collect-calibration` re-renders the same expert state under clean and a distractor condition, using separate native history buffers. It retains identical robot states and expert actions, adds native action normalization, the teacher-forced SimpleMemVLA suffix, query features and source-key alignment, and saves hashed paired samples. It uses the environment renderer, not pixel post-processing.
3. `calibrate` optimizes only W and beta through the native action flow-matching loss. FS uses Adam / 32 updates and SimpleMemVLA AdamW / 16 updates by default, learning rate .003, zero weight decay, gradient norm cap 1. The numerical settings are the recorded RoboMME calibration settings in supplement Table 3, exposed as defaults rather than claimed Arena-specific training measurements.

```bash
robustness_arena_test/.venv-simplemem/bin/python -m robo_ri collect-calibration \
  --method simplememvla --demonstrations configs/calibration_replays.local.json \
  --paths configs/paths.local.json --out calibration/simplememvla --gpu 4,7
robustness_arena_test/.venv-simplemem/bin/python -m robo_ri calibrate \
  --method simplememvla --samples calibration/simplememvla/manifest.json \
  --out robustness_arena_test/submit_codes/checkpoints/simplememvla_ri.npz --gpu 4
```

Use the FS JAX interpreter and `--method fs_mod` for its collection/calibration and save `checkpoints/fs_mod_ri.npz`. All paths in these commands are relative to the current working directory; adjust the local JSON locations accordingly. Calibration rejects test splits, missing clean pairs, action/state mismatches, hash mismatches and absent gradients. It uses the final checkpoint without test-set model selection.

## Results and validation

```bash
robustness_arena_test/.venv-eval/bin/python -m robo_ri report \
  --manifest robustness_arena_test/submit_codes/runs/smoke/manifest.json \
  --out robustness_arena_test/submit_codes/runs/smoke/report.json
```

TSR is the mean of task-level success fractions. Reports expose planned/completed/missing/error/valid counts, per-variant and handin/slide rates. `tsr_full` is null until every planned episode completes with a valid event under the nominal timing. Partial and valid-event estimates have explicit names. Paired timing requires both episodes successful and their initial scene hashes equal. Execution time starts at the first policy step, excluding initialization/settling and post-success monitoring.

`validation/` records unit checks, native attention identity/gradient checks, MuJoCo event mechanics, and a real FS checkpoint smoke run. Synthetic mechanics/attention checks never enter task TSR. Full pretrained RI-Mem training and nine-method paper-scale evaluation have not been performed by this submission preparation; the external Qwen/SigLIP/MemER weights and training demonstrations must be made available at the configured paths. See `validation/README.md` for the exact verified scope.

## Appendix videos

Open `appendix_videos/index.html` or the executed notebook. There are **ten perturbed failure videos and ten matched clean-success videos**. Every pair has the same task, seed, method and initial scene hash; each directory contains original result records, event trace, SHA-256 hashes and a four-panel comparison. `cases.csv` gives a compact inventory and `captions.tex` supplies appendix caption text. `tools/build_appendix.py` regenerates the package from the existing `../results` files; it never labels a successful episode as a failure or generates substitute model footage.

These are historical recordings with the same ten event families. Several old parameters differ from supplement Table 2 (e.g. longer camera/light events, larger shift and longer remove/return). Every `case.json` explicitly sets `paper_parameter_match: false`. They support qualitative illustration, not an exact-parameter replication claim. The ten selected cases use FS+Mod so method differences do not confound the appendix gallery. Do not infer population TSR from this deliberately selected failure set.

Attribution and licenses are recorded in `THIRD_PARTY.md`.
