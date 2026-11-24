# Sources and integration boundaries

This folder adds an Arena-only experiment layer and RI-Mem. Model implementations, pretrained weights, simulators and training demonstrations remain in their source repositories and retain their original licenses. No ownership or relicensing of those assets is implied.

| Source | Local integration |
|---|---|
| RoboMemArena, `RoboMemArena/` | Native BDDL scenes, observation conversion, stage scorer and asynchronous planner/rollout. The local harness pins revision `d09f63b671f362985e1e0d038d2014c2d2519488`. |
| MME-VLA, `../third_party/robomme-mme-vla-all26-eval/` and `../../robomme_policy_learning/` | Released all-26 FS+Mod model, memory buffer and challenge interface. Apache-2.0 source; Gemma-derived assets also retain the source repository's `LICENSE_GEMMA.txt`. `ri_mem/bindings.py` extends loaded modules in memory and checks source anchors; upstream files and pretrained parameter names are unchanged. |
| SimpleMemVLA, `../third_party/SimpleMemVLA/` | Native Arena history construction, processor, action objective and adapter. MIT, copyright 2025 Cheng Yin; see that repository's LICENSE. |
| TraceFlow, `../third_party/TraceFlow/` | Native remote planner clients, upper/lower services and suite-specific retrieval heads/gates. Apache-2.0; see its LICENSE. `tools/serve_traceflow.py` derives a launcher snapshot and replaces the environment-pool invocation with a service wait. |
| PrediMem, provided within RoboMemArena | Native asynchronous keyframe planner and pi0.5 action checkpoint, preserving native action normalization and scoring. See the source repository/model-card terms. |
| MemER, `robomme_policy_learning/examples/robomme/subgoal_prediction/qwenvl/api_memer.py` | Official predictor with boundary handling for short histories and empty keyframe sets; runs over Arena observations and the separately configured MemER adapter. |
| RoboMME-Interference, SoumilRathi/robomme-interference, revision `3536f63bbd483654cdd50ef86a656572a03fb72d` | `robo_ri/retrieval.py` adapts its visual segment gate to causal streaming Arena memory: adjacent cosine cuts at .923, minimum segment four, select highest query-frame similarity segment, native uniform frame budget. Current query is excluded from retrieval candidates. The original local source is `robustness_test/third_party/robomme-interference/retrieval/robomme_visual_gate.py`; its repository LICENSE is Apache-2.0. |
| Qwen3-VL-4B-Instruct and SigLIP SO400M/14/384 | Frozen auxiliary perception/planning/retrieval models. Weights are external and not distributed here; preserve model-card license and attribution. |

The supplied PDFs define the protocol and RI-Mem equations. They do not include an Arena detector checkpoint or serialized RI-Mem calibration weights. The sparse VLM region grounding/template tracking implementation and explicit native attention adapters are implementation choices documented in README.md, not claimed recovered author code.

Historical appendix videos and original result/scenario JSON are copied from the user's local Arena experiments. Their source paths and SHA-256 hashes are included with each case. They contain no newly generated model actions or success labels.
