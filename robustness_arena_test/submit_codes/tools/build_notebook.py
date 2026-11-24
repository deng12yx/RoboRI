from pathlib import Path
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = [nbf.v4.new_markdown_cell("""# RoboMemArena-RI：clean 成功与扰动后失败

本 notebook 展示 10 种扰动的真实录像。每个案例均为 FS+Mod，同任务、同 seed、初始场景哈希一致，clean 成功而扰动后失败。

**样本用途**：这些是已有 Arena 实验的定性案例，事件名称映射到本文的 10 类。旧实验的位移、触发时刻、持续时间与论文补充材料 Table 2 不完全一致，不能作为新提交代码默认参数下的复现实测。这里展示的是挑选的失败样本，不用于估计总体成功率。

左视频为 clean，右视频为扰动后失败。图片和指标保存在 notebook 中；原始视频通过同目录 `appendix_videos` 的相对路径读取。"""),
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
print("已核验 10 对录像：clean 成功，扰动后失败，初始场景相同，视频 SHA-256 一致。")'''),
nbf.v4.new_markdown_cell("""## 实验代码的分类与汇总

`robo_ri/protocol.py` 定义 4 类任务物体扰动与 6 类干扰事件；clean 是独立对照条件。late 类别不进入矩阵。

7 个 baseline 为 FS+Mod、SimpleMemVLA、TraceFlow、MemER、PrediMem、FS+Mod (Retrieval)、π0.5。另有 FS+Mod + RI-Mem 与 SimpleMemVLA + RI-Mem。

正式汇总采用任务宏平均。完整率、无效事件、技术错误和缺失 episode 单独报告；配对耗时仅统计 clean 与扰动均成功且场景哈希相同的样本。"""),
nbf.v4.new_code_cell('''from robo_ri.protocol import CONDITIONS, METHODS
rows = "".join(f"<tr><td>{name}</td><td>{c.family}</td><td>{', '.join(c.variants)}</td></tr>" for name, c in CONDITIONS.items())
display(HTML("<table><tr><th>Condition</th><th>Family</th><th>Variants</th></tr>" + rows + "</table>"))
print("Baseline 和增强版本：", ", ".join(METHODS))'''),
nbf.v4.new_markdown_cell(r"""## 统一的 RI-Mem 读取增强

SCQ 将目标、当前子目标和证据需求编码成 6 维查询。AMR 使用相关性、明确无关性、当前可观测性与时间有效性，生成每个历史 key 的有界注意力偏置和共享门控。可训练参数只有 $W\in\mathbb{R}^{6\times5}$ 与 $\beta\in\mathbb{R}^{5}$，共 35 个。

FS+Mod 在专用 memory attention 中添加偏置并缩放读取结果；SimpleMemVLA 在第 27、31 个 full-attention 层为历史 video key 添加偏置与 $\log g$。两者保留各自的记忆选择和动作预测结构。下面只检查公式与接口，不把这些数值当作任务评测。"""),
nbf.v4.new_code_cell('''import numpy as np
from robo_ri.ri_mem.head import initial_parameters, numpy_priors
p = initial_parameters()
query = np.array([1, 1, 0, 0, 1, 1], np.float32)
features = np.array([[.9, 0, .8, -1], [0, .9, 0, 0], [.9, 0, 0, 1]], np.float32)
bias, gate = numpy_priors(p, query, features)
assert sum(value.size for value in p.values()) == 35
assert (np.abs(bias) <= .6).all() and .92 <= gate <= 1.08
print("示例 key 偏置：", bias.round(4).tolist(), "门控：", round(float(gate), 4))''')]

for i in range(10):
    cells.append(nbf.v4.new_code_cell(f'''case = cases[{i}]
display(Markdown(f"## {{case['number']:02d}}. {{case['condition']}} — Task {{case['task']}}, seed {{case['seed']}}"))
display(Markdown(case["prompt"]))
display(Image(filename=str(ROOT / "appendix_videos" / case["comparison"])))
clean_url = "appendix_videos/" + case["clean_video"]
failure_url = "appendix_videos/" + case["failure_video"]
display(HTML(f'<video controls preload="none" width="49%" src="{{clean_url}}"></video> <video controls preload="none" width="49%" src="{{failure_url}}"></video>'))
print("未完成的 native stages：", ", ".join(case["incomplete_stages"]))
print("源条件：", case["original_condition"], "| 录像中的事件帧：", case["video_event_frame_range"])
print("论文默认参数匹配：", case["paper_parameter_match"], "| 协议：", case["source_protocol"])
'''))
cells.append(nbf.v4.new_markdown_cell("""## 运行与结果边界

完整命令与环境见 `README.md`。可通过 `python -m robo_ri plan` → `serve-fs` / 原生后端 → `run` → `report` 完成实验闭环。

本提交中的 `validation` 保存机械触发和接口验证；`appendix_videos` 保存历史实测案例。新 RI-Mem checkpoint 需要使用训练集上的 clean/扰动配对专家样本校准后生成；没有把论文表格中的成功率写成新实验结果。"""))
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "RoboMemArena RI", "language": "python", "name": "robo-ri-eval"},
                    "language_info": {"name": "python", "version": "3.11"}}
nbf.write(nb, ROOT / "robomemarena_ri_examples.ipynb")
print(ROOT / "robomemarena_ri_examples.ipynb")
