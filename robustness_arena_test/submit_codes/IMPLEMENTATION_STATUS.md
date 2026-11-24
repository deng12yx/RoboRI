# Arena implementation coverage

The main paper's RoboMemArena results table and supplement section 2.2 specify seven baselines. Supplement section 2.3 applies RI-Mem to FS+Mod and SimpleMemVLA on Arena; TokenDrop+Mod is evaluated only on RoboMME. No Arena TokenDrop baseline or ablation matrix is added here.

## Baselines and perturbations

All seven baselines use all ten conditions below, with a separate clean control:

`onset_arrival`, `mid_arrival`, `mid_shift`, `remove_return`, `tv_on`, `object_passby`, `unrelated_object`, `target_like_object`, `camera_shake`, `light_change`.

| Paper method | Method ID | Native execution | Ten conditions configured |
| --- | --- | --- | --- |
| FS+Mod | `fs_mod` | `FrameSample` client and native FS server | Yes |
| FS+Mod (Retrieval) | `fs_mod_retrieval` | Same native policy with causal SigLIP retrieval | Yes |
| SimpleMemVLA | `simplememvla` | Native Arena policy adapter | Yes |
| TraceFlow | `traceflow` | Native asynchronous upper/lower services | Yes |
| MemER | `memer` | Official subgoal predictor and Arena pi0.5 action policy | Yes |
| PrediMem | `predimem` | Native asynchronous memory planner and pi0.5 action policy | Yes |
| pi0.5 | `pi05` | Native action server with the task instruction | Yes |

`runner.run_episode` installs `Perturbation` before either execution path starts. Generic adapters and the native PrediMem/TraceFlow loops receive the same wrapped environment. Events advance on every executed control step, including actions inside a predicted chunk. Model inputs retain the native observation interface; event labels, stage state, and simulator object poses are not supplied to the policy.

`perturbations.py` implements the ten events in MuJoCo. It records onset and completion, camera visibility, manipulation boundaries, and validity. An episode without a feasible or completed event is reported as invalid; it is not silently treated as a completed perturbation. Late-arrival variants are rejected. Arrival presentation style and displacement/distractor subtypes are deterministic across methods for a matched task and seed.

`configs/paper.json` contains the complete Cartesian product: 26 tasks, 51 seeds, clean plus ten conditions, seven baselines plus two RI-Mem variants. This is a planned evaluation matrix, not a completed benchmark.

## RI-Mem implementations

| Arena variant | Readout adaptation | Preserved native components |
| --- | --- | --- |
| `fs_mod_ri` | Key biases in dedicated memory attention; gate on its output | Frame selection, 512-key budget, stored features, action generation |
| `simplememvla_ri` | Historical video-key biases and log gate in full-attention layers 27 and 31 | Video history selection, temporal mapping, KV caches, action generation |

Both use the shared SCQ, observation-derived evidence index, bounded AMR priors, and a separate 35-parameter calibration head. The paired expert replay collector, frozen-backbone native action-loss calibration, inference bindings, and evaluation/reporting commands are implemented. Neither variant substitutes initialized parameters for a required calibrated checkpoint.

The sparse VLM grounding and template tracker are documented implementation choices because the PDFs do not provide an Arena detector checkpoint. They are not claimed to be recovered author code.

## Verified scope

- The configuration audit checks all 70 baseline/perturbation combinations, both Arena RI-Mem variants, and English source comments/docstrings/notebook explanations.
- Existing recorded checks cover all ten real MuJoCo event mechanics, additional removal/dark subtypes, and all 26 task profiles.
- Native FS and SimpleMem attention checks cover readout identity and gradient propagation. The SimpleMem training check uses a frozen action-policy fixture rather than a pretrained VLA.
- The recorded pretrained FS+Mod smoke run covers clean, target shift, and light change on one task/seed. The twenty appendix videos are historical matched examples.

The other pretrained baseline rollouts and full RI-Mem calibration have not all been executed during submission preparation. External native repositories, compatible environments, model weights, and official training demonstrations remain required. In particular, MemER needs an explicit Arena adapter; the code does not replace it with a RoboMME checkpoint. Code and configuration coverage must not be described as a completed seven-method, ten-condition experimental result.

Run the audit from the project root:

```bash
robustness_arena_test/.venv-eval/bin/python \
  robustness_arena_test/submit_codes/tools/check_coverage.py
```
