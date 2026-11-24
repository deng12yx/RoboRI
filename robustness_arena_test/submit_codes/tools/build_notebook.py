from pathlib import Path
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = [nbf.v4.new_markdown_cell("""# RoboMemArena-RI: Clean Success and Perturbed Failure

This notebook presents recorded failures under ten perturbation types. Each FS+Mod case is paired with a clean success using the same task, seed, and initial scene hash.

**Scope.** These historical Arena recordings illustrate the ten event families. Their displacement, timing, and duration can differ from supplementary Table 2. They are selected qualitative examples, not measurements under the submission defaults or estimates of population success rates.

The clean video appears on the left and the perturbed failure on the right. Comparison images and metrics are embedded; videos use relative paths into `appendix_videos`."""),
nbf.v4.new_code_cell('''from pathlib import Path
import hashlib
import json
from IPython.display import display, Image, HTML, Markdown

ROOT = Path.cwd()
if not (ROOT / "appendix_videos/manifest.json").exists():
    ROOT = ROOT / "robustness_arena_test/submit_codes"
gallery = json.loads((ROOT / "appendix_videos/manifest.json").read_text())
cases = gallery["cases"]
assert len(cases) == 10
for case in cases:
    directory = ROOT / "appendix_videos" / Path(case["failure_video"]).parent
    clean = json.loads((directory / "clean_result.json").read_text())
    failure = json.loads((directory / "failure_result.json").read_text())
    assert clean["tsr"] == 1 and failure["tsr"] == 0
    assert clean["task"] == failure["task"] and clean["seed"] == failure["seed"]
    assert clean["init_state_sha1"] == failure["init_state_sha1"]
    for role in ("failure", "clean"):
        video = ROOT / "appendix_videos" / case[f"{role}_video"]
        assert hashlib.sha256(video.read_bytes()).hexdigest() == case[f"{role}_sha256"]
print("Verified ten clean-success / perturbed-failure pairs, matching initial scenes and video SHA-256 hashes.")'''),
nbf.v4.new_markdown_cell("""## Conditions and Reporting

`robo_ri/protocol.py` defines four task-object perturbations and six distractor events. Clean is a separate control. Late-arrival variants are excluded.

The seven baselines are FS+Mod, SimpleMemVLA, TraceFlow, MemER, PrediMem, FS+Mod (Retrieval), and pi0.5. The paper's Arena RI-Mem variants use FS+Mod and SimpleMemVLA.

Success rates use a task macro-average. Coverage, invalid events, technical errors, and missing episodes are reported separately. Timing comparisons require both paired episodes to succeed with the same initial scene hash."""),
nbf.v4.new_code_cell('''from robo_ri.protocol import CONDITIONS, METHODS
rows = "".join(f"<tr><td>{name}</td><td>{c.family}</td><td>{', '.join(c.variants)}</td></tr>" for name, c in CONDITIONS.items())
display(HTML("<table><tr><th>Condition</th><th>Family</th><th>Variants</th></tr>" + rows + "</table>"))
print("Registered methods:", ", ".join(METHODS))'''),
nbf.v4.new_markdown_cell(r"""## Shared RI-Mem Readout

SCQ encodes the active subgoal and evidence requirements as a six-dimensional query. AMR uses relevance, explicit irrelevance, observability, and temporal validity to produce bounded historical-key biases and a shared gate. Only $W\in\mathbb{R}^{6\times5}$ and $\beta\in\mathbb{R}^{5}$ are trainable: 35 parameters in total.

FS+Mod adds biases to dedicated memory attention and gates its output. SimpleMemVLA applies biases and $\log g$ to historical video keys in full-attention layers 27 and 31. Both preserve native memory selection and action prediction. The following numerical check is not a task evaluation."""),
nbf.v4.new_code_cell('''import numpy as np
from robo_ri.ri_mem.head import initial_parameters, numpy_priors
p = initial_parameters()
query = np.array([1, 1, 0, 0, 1, 1], np.float32)
features = np.array([[.9, 0, .8, -1], [0, .9, 0, 0], [.9, 0, 0, 1]], np.float32)
bias, gate = numpy_priors(p, query, features)
assert sum(value.size for value in p.values()) == 35
assert (np.abs(bias) <= .6).all() and .92 <= gate <= 1.08
print("Example key biases:", bias.round(4).tolist(), "gate:", round(float(gate), 4))''')]

for i in range(10):
    cells.append(nbf.v4.new_code_cell(f'''case = cases[{i}]
display(Markdown(f"## {{case['number']:02d}}. {{case['condition']}} — Task {{case['task']}}, seed {{case['seed']}}"))
display(Markdown(case["prompt"]))
display(Image(filename=str(ROOT / "appendix_videos" / case["comparison"])))
clean_url = "appendix_videos/" + case["clean_video"]
failure_url = "appendix_videos/" + case["failure_video"]
display(HTML(f'<video controls preload="none" width="49%" src="{{clean_url}}"></video> <video controls preload="none" width="49%" src="{{failure_url}}"></video>'))
print("Incomplete native stages:", ", ".join(case["incomplete_stages"]))
print("Source condition:", case["original_condition"], "| Event video frames:", case["video_event_frame_range"])
print("Paper-default parameter match:", case["paper_parameter_match"], "| Protocol:", case["source_protocol"])
'''))
cells.append(nbf.v4.new_markdown_cell("""## Execution and Validation Scope

Commands and environment requirements are documented in `README.md`. The evaluation workflow is `python -m robo_ri plan`, start the native backend, then `run` and `report`.

`validation` records mechanics and interface checks; `appendix_videos` contains historical measured cases. RI-Mem checkpoints require calibration on paired clean and perturbed expert training samples. Paper table entries are not treated as newly measured results."""))
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "RoboMemArena RI", "language": "python", "name": "robo-ri-eval"},
                    "language_info": {"name": "python", "version": "3.11"}}
nbf.write(nb, ROOT / "robomemarena_ri_examples.ipynb")
print(ROOT / "robomemarena_ri_examples.ipynb")
