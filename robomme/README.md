# RoboMME-RI 与 RI-Mem

分支：`robomme-submit`。本模块放在 `robomme/`，与 `arena/ri-mem-submission-20261124` 分支中的 `robustness_arena_test/` 分开维护。

范围为 RoboMME。协议以 main paper（Robo_RI_CVPR27.pdf） §3、§5 和 Table 2 为准；RI-Mem 细节对应 supplement（Robo_RI_CVPR27_supplement.pdf） §1、§2.3、Tables 2–3。没有包含 RoboMemArena、RoboTwin 的适配代码。

## 目录

| 路径 | 内容 |
|---|---|
| `robomme_ri/protocol.py` | 16 个任务、10 个扰动条件及确定性 seed |
| `perturbations.py`、`scene.py` | 控制步事件、ManiSkill 场景操作及恢复 |
| `baselines.py`、`policies.py`、`official.py` | baseline 注册表及本地/服务端适配 |
| `rimem/` | SCQ、证据索引、AMR、三个 backbone 的 attention 接入 |
| `calibrate.py` | 冻结 backbone、使用原生 flow-matching loss 校准 35 参数 head |
| `evaluate.py` | 观测 → 推理 → 执行动作 → 新观测的闭环；结果和事件日志 |
| `configs/experiment.json` | 17 个方法版本 × 4 任务 × 3 seed × 11 条件，2,244 episodes |
| `configs/paper.json` | 28 个 baseline + 3 个 RI-Mem 版本的完整实验网格 |
| `configs/history.json` | 三种主方法的 No History、Shuffled History、OracleMem 对照 |
| `notebooks/robomme_ri_demo.ipynb` | 3 个方法、4 组真实配对案例与多 baseline 实验入口 |

## 运行

```bash
cd robomme
python -m pip install -e '.[notebook]'
export LAPSE_ROOT=/path/to/LAPSE
export MODEL_ROOT=/path/to/mme_vla_suite
export QWEN_MODEL=/path/to/Qwen3-VL-4B-Instruct
python -m robomme_ri.cli inventory
python -m robomme_ri.cli plan configs/experiment.json --output outputs/plan
python scripts/run_experiment.py configs/experiment.json --output outputs/experiment --gpus 0,1
```

`LAPSE_ROOT` 指向包含上游 RoboMME 模型与环境源码的工作目录，并非本仓库的 clone 目录。在 JSON 中填写各方法的 Python 环境、checkpoint、LoRA 与服务地址。模型源码使用原有官方仓库；不重复打包权重和上游训练框架。`run_experiment.py` 为每个方法启动独立进程，并设置可见 GPU。远程 baseline 服务应在同一配置指定的设备启动，客户端 GPU 设置不会迁移已有服务。

也可以在方法配置中填写 `server_commands`（命令参数列表组成的列表）。运行器会启动这些服务、等待配置中的端口就绪，结束或中断时关闭本次启动的进程。命令参数中的 `{gpu}` 替换为分配的设备编号。

单方法调试：

```bash
python -m robomme_ri.cli evaluate configs/experiment.json \
  --method framesamp-modul --output outputs/fs
python -m robomme_ri.cli summarize outputs/fs/shard_0/rollouts.jsonl \
  --output outputs/fs/summary.json
```

`--shard k/n` 以 `(task, episode)` 分片，同一对 clean/perturbed 保持在同一分片。相同配置的已完成 episode 可续跑。技术错误允许重试；事件没有执行的样本保留独立状态。

## 扰动

| 大类 | 条件 | RoboMME 默认值 |
|---|---|---|
| Remember | Onset Arrival | native 初始化阶段进入，水平偏移 0.65 m，最高偏移 0.20 m，n=20 前落定 |
| Remember | Mid-task Arrival | 后续阶段对象提前进入，0.65 m，最高 0.25 m，最多 12 步 |
| Remember | Slight Shift | t≥40，水平 0.02 m；四方向中选择对象间距最大的可行终点 |
| Remember | Remove & Return | t=40 移出，偏移 (0,0.65,0.25) m，最多 12 步，使用前恢复 |
| Ignore | TV On | t=16–96，四个彩块，每 8 步更新 |
| Ignore | Object Pass-by | t=32–96，背景路径 0.68 m |
| Ignore | Unrelated Object | 无碰撞物件，t=16 插入或 t=48 移除 |
| Ignore | Target-like Object | 5 cm 绿块、白十字；同上 |
| Ignore | Camera Shake | t=16，振幅系数 0.018 m，3 步后恢复 |
| Ignore | Light Change | t=16，ambient=(2,2,2) 或 (0,0,0)，3 步后恢复 |

插入/移除、明/暗各按 episode 奇偶均衡分配，也可在 `EpisodeSpec` 中显式指定。十个顶层条件分别评测；不将旧脚本的 onset/mid 随机混合结果当作这十个条件。

事件挂在 `BaseEnv.step`，waypoint 规划器和动作块内部的每个控制步都会推进。native 初始化和 online 执行有各自时钟。事件不增加固定等待、不补预算、不改变指令和成功判据。Shift 保留新终点，使用前要求可见、可达；临时移除恢复原位。

任务对象选择器采取保守约束：不搬动正在抓取的物件，不抬起盖住答案的容器；不修改 StopCube 的任务时钟。PatternLock/RouteStick 从任务闭包解析目标，仍须满足几何与阶段约束。Shift 和 Remove & Return 可以选择同类的非答案对象，并记录 target/nontarget；Mid Arrival 选择后续需要操作的对象。若物件需要使用时事件仍未完成，该事件标为 invalid 并恢复。无法找到合规窗口的 task-object episode 记录为 `inapplicable`。汇总报告实际覆盖率；生成 8,000 个扰动任务的计划不等于获得 8,000 个有效样本。特殊任务的专用窗口仍需进一步验证。

## Baseline

主实验注册表包含 28 个 baseline 版本及三个 RI-Mem 版本；另有九个历史干预对照版本。

| 类别 | 方法 |
|---|---|
| Perceptual | FS/TD × Ctx/Mod/Exp；OFT FS/TD+Ctx；FS+Mod Retrieval；SimpleMemVLA；SAM2Act+；HAMLET |
| Symbolic | SimpleSG/GroundSG × Qwen/Oracle；OFT SimpleSG/GroundSG+Qwen |
| Perceptual-cognitive | MemoryVLA、MemER、PonderPounce |
| Recurrent | TTT+Ctx、TTT+Exp |
| Memory-free | π0.5、SAM2Act、Diffusion Policy；另保留 §5.1 提及的 SmolVLA、OpenVLA-OFT |

MME 和 SimpleMemVLA 直接调用原生模型。Qwen baseline 使用官方方法对应的 LoRA，RI-Mem 的 SCQ 使用冻结的基础 Qwen3-VL。Oracle 是独立命名的 baseline，不会自动替代 Qwen。

Qwen/LoRA 可以与 MME 分进程运行：使用 `.venv-vlm/bin/python scripts/serve_subgoals.py --native symbolic-simple-subgoal --adapter ADAPTER --qwen-model MODEL --port 8891` 启动服务，在对应方法配置填写 `subgoal_endpoint: ws://127.0.0.1:8891`。这保留官方 predictor 的历史对话与 MemER 关键帧逻辑。

其他方法通过官方服务运行：OFT/SmolVLA 使用 HTTP；SAM2Act、DP、MemoryVLA 使用已有发布适配的 WebSocket 接口；HAMLET 使用官方 ZMQ。SAM2Act 保留 40 waypoint 和原生物理步预算，MemoryVLA 保留 EE delta 转换及 8 步执行。它们与 joint-angle 方法共用事件和结果逻辑。

PonderPounce 的官方源码/权重不在当前仓库。已提供标准 RoboMME challenge 服务客户端；需要将其官方策略通过 `robomme-challenge-v1` 暴露，或在配置填写 `factory: module:function`。该项尚未验证官方服务兼容性，不能将注册表存在等同于该模型已经跑通。

## RI-Mem

三种 backbone 共用 `RIMem`：任务分解 → 异步进度检查 → RGB 证据索引 → retained-key 对齐 → 35 参数 head → attention readout。

- 查询为 `[S,I,P,M,c,1]`。每四次决策最多提交一次检查，每 episode 最多 16 次；完成置信度 ≥0.85 才推进一个阶段。阶段前 20 次决策 `c=1`，之后 `c=.5`。
- 证据特征依照 supplement Eqs.2–7；只对显式识别的 distractor 施加 irrelevance 惩罚。历史坐标与当前对象不匹配时 spatial validity 为 −1，identity/motion 不因位置变化失效。
- `bias=.6*tanh(p/.6)`；`gate=1+.08*tanh(y5)`。低阶段置信度只保留 irrelevance 惩罚；空证据时 gate=1。
- FS+Mod / TD+Mod 保持 512 keys 及原选择机制，在原 memory attention 中加 bias，输出乘 gate。
- SimpleMemVLA 仅接入 full-attention 层 27、31；历史视频 keys 加 bias 与 log(gate)，其他 keys、mask、KV cache 和动作预测保持原有路径。推理对时间合并 key 平均 source-frame 的有界 prior。

HSV 检测和白十字规则集中于 `rimem/perception.py`。无法从 RGB 确认的语义不产生虚构证据；此简洁检测器的覆盖范围需要用实际任务图像进一步校准。证据索引不读取模拟器答案、事件对象标签或未来帧。

### 校准

每个 backbone 使用独立 head。MME：32 decision points、32 updates、Adam；SimpleMemVLA：8 decision points、16 updates、AdamW。均为 batch=1、lr=0.003、零 weight decay、梯度裁剪 1，保持模型冻结。初始化和其他参数对应 supplement §2.3。

```bash
python -m robomme_ri.cli calibrate configs/experiment.json \
  --backbone framesamp-modul --manifest calibration/fs/manifest.json \
  --output checkpoints/framesamp-modul.npz
```

manifest 为 JSON 列表；每行包含 `split=train`、`query_source=expert_subtask`、`augmentation=environment_render`、`pair_id`、`variant=clean|perturbed`、`robot_state_sha256`、`actions_sha256` 和相对 `batch` 路径。同一 pair 的机器人状态及 expert action 必须一致。NPZ 不使用 pickle。

`calibration_data.CalibrationWriter` 写入成对 batch 及原始机器人状态、expert actions；加载时重新计算 hash 校验。输入 batch 仍需由官方 expert 训练轨迹和环境渲染得到。

MME batch 包含 `query:[6]`、`features:[512,4]`、原生归一化的 `actions`，以及 `observation/` 前缀展平的 `HistAugObservation.to_dict()`。SimpleMemVLA batch 包含相同 query/features 和 `video_positions`，其余字段为原生 `forward` 的 teacher-forced 输入：视频/图像、input_ids、attention_mask、expert subtask labels、归一化 actions、action_mask、state。时间对的 evidence features 在训练前平均。

当前 SimpleMemVLA 校准使用原生完整 teacher-forced forward；论文描述的缓存 prefix 校准需要由上游数据导出器提供对应缓存路径，尚未在此提交中复现其吞吐。没有随包提供已校准权重，加载器会拒绝将初始化 head 用作正式增强实验。

11 项 ablation 入口见 `baselines.ABLATIONS`。`reweight_only` 使用总体任务实体；fixed/random weights 的默认值在 `head.py` 中显式指定，论文没有给出这两个对照的完整采样细节，因此不能将这些默认设置声称为原始实验参数。

`python scripts/run_ablations.py configs/paper.json --output outputs/ablations` 生成 11 份配置；增加 `--run --gpus 0,1` 后运行。No History、Shuffled History、OracleMem 的接口与参数见 `history_controls.py`；OracleMem 独立读取特权分割，仅用于该对照。

## Notebook 与验证

Notebook 的 4 对视频和静态帧已复制到 `notebooks/assets`，可脱离原结果目录播放。每个案例保存原始行号、记录和 SHA256，包含 instruction。示例来自历史实验；旧时序、幅度和预算与新版论文协议不同。增强后的效果须用新评测结果补充。

本分支只包含 notebook 引用的四组视频和关键帧，见 [案例索引](notebooks/README.md)。Shift 示例实际移动 44.8 mm，与论文默认 20 mm 不同。另外六类 Ignore 视频尚未包含。

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu /path/to/mme-python -m pytest tests -q
/path/to/simplememvla-python scripts/check_simplememvla_attention.py
CUDA_VISIBLE_DEVICES=0 /path/to/mme-python scripts/smoke_simulator.py
python scripts/execute_notebook.py
```

`smoke_simulator.py` 使用 stationary controller 验证场景事件，不产生 baseline 成功率。新模型实验必须通过 `evaluate` 运行；旧视频、仿真接口测试和完整任务评测保持独立记录。
