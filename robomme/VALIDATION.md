# Validation status

The local submission assembly passed the following checks on 2026-11-24:

- 26 unit tests covering protocol handling, evidence scoring, history controls and memory attention.
- Native SimpleMemVLA attention-dispatch checks, including layer scope, gradient propagation and restoration.
- Simulator interface checks for clean and all ten perturbation conditions. These used a stationary controller and do not measure baseline task success.
- Process-level planning for the 17 method versions in `configs/experiment.json`.
- Execution of all nine notebook code cells, including execution from an extracted submission archive independent of the original result directory.
- Hash verification of the four notebook videos against `notebooks/cases.json`.

Notebook outputs are cleared in Git to avoid duplicating the MP4 data. Run `scripts/execute_notebook.py` in an environment with the notebook dependencies to display them.

## Remaining reproduction requirements

Full pretrained-baseline closed-loop evaluation and RI-Mem calibration require the upstream checkpoints and expert calibration batches. Learned RI-Mem heads are not included. The PonderPounce official service interface has not been verified. Full task/seed admissibility coverage still requires validation, and SimpleMemVLA cached-prefix calibration throughput has not been reproduced.

## Qualitative examples

All four videos are matched historical clean-success / perturbed-failure pairs. Their event timing, displacement and budgets differ from the paper protocol; they are qualitative examples rather than new-protocol success-rate measurements. The shift example moves the object by 44.8 mm; the paper default is 20 mm. Six Ignore-category example videos are not included.
