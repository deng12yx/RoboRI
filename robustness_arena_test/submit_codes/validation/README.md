# Validation scope

The artifacts in this directory have different purposes; they must not be pooled into a benchmark TSR.

| Artifact | What it verifies |
|---|---|
| `checks.json` | Recorded software checks: 13 unit tests, native attention integration, task catalog, executed notebook. |
| `physics/summary.json` | All ten event types in a real MuJoCo Arena task-25 environment. Scripted stage/pose triggers exercise the event mechanics; no learned policy is used. |
| `physics_seed51/summary.json` | Additional removal subtypes and dark illumination variant. |
| `catalog.json` | All 26 profiles' object names resolve in native BDDL and each task has a native scorer. |
| `artifacts.json` | Ten strict clean-success/failure pairs; SHA-256 checked; all twenty videos decoded completely; all notebook code cells executed with ten embedded images. |
| `fs_checkpoint_smoke_final/` | Real pretrained FS+Mod + native Arena rollout/scorer, task 25, seed 51, clean / shift / light. A small pipeline check, not a population estimate. |
| `plan_check_final/` | Planned small matrix and its empty coverage report; missing episodes deliberately have null complete TSR. |

The FS attention check uses the released JAX memory-attention implementation and a native scanned Gemma module. It checks zero-prior equivalence, nonzero gradients through key bias, and bfloat16 output preservation. The SimpleMemVLA attention check uses native Transformers Qwen3.5 full-attention layers 27/31, a small frozen action-policy fixture, and a two-update 35-parameter head calibration; it verifies that backbone parameters remain unchanged. It does not load a pretrained SimpleMemVLA policy or claim a trained RI-Mem success rate.

The full RI-Mem calibration data collector, pretrained SimpleMemVLA/RI inference, MemER, TraceFlow, PrediMem, and SigLIP retrieval require their external model/training assets. Their full pretrained rollouts were not all rerun during submission preparation. The Qwen/SigLIP shared-model mount was unavailable during the initial checks. No trained RI-Mem checkpoint or full paper table is claimed here.

Re-run checks from the LAPSE root:

```bash
export PYTHONPATH="$PWD/robustness_arena_test/submit_codes"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 robustness_arena_test/.venv-eval/bin/python \
  -m pytest robustness_arena_test/submit_codes/tests -q
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  robomme_policy_learning/.venv/bin/python robustness_arena_test/submit_codes/tools/check_native_fs.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 robustness_arena_test/.venv-simplemem/bin/python \
  robustness_arena_test/submit_codes/tools/check_native_simplemem.py
robustness_arena_test/.venv-eval/bin/python robustness_arena_test/submit_codes/tools/check_physics.py \
  --gpu 4 --out robustness_arena_test/submit_codes/validation/physics
robustness_arena_test/.venv-eval/bin/python robustness_arena_test/submit_codes/tools/check_catalog.py
.venv/bin/python robustness_arena_test/submit_codes/tools/execute_notebook.py
robustness_arena_test/.venv-eval/bin/python robustness_arena_test/submit_codes/tools/verify_artifacts.py
```

`validation/physics` deliberately records the original sampled scene hash before interventions. All task-25 conditions use the same underlying reset state. The checkpoint smoke records actual interaction timings; a target shift must finish before the recorded target manipulation, not merely before task termination.
