# CS336 Assignment 2 — Writeup

Systems and Parallelism · Spring 2026 · Handout Version 26.1.3

> 本模板依据本地英文讲义整理。题目为中文摘要，不是逐字翻译；算法、公式、示例代码和接口细节请对照所标页码。本次依据所附 Part 2 讲义（原页码 2–10）补写第 2 章；后续章节保留原模板。
>
> 本次讲义：`cs336_assignment2_systems_part2(1).pdf`（第 2 章及第 3 章开头）。最终提交要求为 `writeup.pdf` 和 `code.zip`；本文件是报告的 Markdown 编辑源。
>
> 表格可自行扩展，示例行不代表仅需运行这些配置。未运行、OOM、运行失败请分别注明，不要用 0 代替。实现题的答案栏用于记录代码位置和验证结果，不能替代代码提交。
>
> 对应范围：全部 27 道计分 Problem、58 个作答单元（带字母子题分别计数，无字母题各计一个），另附 1 个可选 Triton 反向记录区。环境信息、日志字段和辅助表格是模板补充，不是新增评分题。末尾提供逐题核对表。


### 本次填写范围与证据状态

本稿保留原模板中的实验记录，新增分析均基于所附 `nsys_profile_tables.md` 和讲义；没有重新运行远端 GPU 实验。六份 `.nsys-rep` 已提供，但当前环境没有 Nsight Systems，无法重新导出或核验原始 kernel 归属，因此下面明确区分“汇总表记录”“理论推导”和“待补测”。本次也未附 `benchmark.py` 或模型源码，原模板中的代码路径和运行记录仅作为已有记录保留，不表示本次已经验证实现。

| 项目 | 本次状态 |
| --- | --- |
| benchmarking_script | 补齐已有计时数据分析；(c) 仅有 small，其他模型的预热对照尚缺 |
| nsys_profile (a)–(e) | 六组均整理；CPU/GPU 归属、forward+backward 独立统计及 inference 对照仍需补核验 |
| mixed_precision_accumulation | 给出数值与误差分析；NumPy 同精度逐次累加复核，待在 PyTorch 执行原代码 |
| benchmarking_mixed_precision (a)(b) | 完成理论回答；(c) 缺 BF16 实测 |
| memory_profiling | 完成 (d) 推导，补充 (f) 理论核算；缺指定配置显存数据及截图 |
| 第 3–9 章 | 本次附件未提供相应讲义正文和实验数据，保留原答案区 |

下文“待补测”表示当前附件无法支持结论，不表示耗时或显存为零；本稿尚不是所有实验完成后的提交终稿。

## 1. 基本信息与实验环境

第 1 章没有单独计分题。

- 姓名 / 学号：【待填写】
- 日期 / 提交版本：2026-09-16；Part 2 数据整理与分析稿（尚缺补测）。
- 代码 commit：【待填写】

| 项目 | 记录 |
| --- | --- |
| 当前可用 GPU（用户提供） | 1 × NVIDIA RTX PRO 6000 Blackwell Server Edition（约 97887 MiB / 96 GB） |
| 各实验实际 GPU / 数量 | AutoDL 单卡 NVIDIA RTX PRO 6000 Blackwell：small/medium/large/xl 分段计时成功；**10B 在默认 batch 4、context 512、FP32 train 下 OOM**（2026-09-16）。 |
| CPU / 主机内存 / 操作系统 | AutoDL：25 核 CPU，120 GB 内存，Ubuntu 22.04。 |
| 驱动 / CUDA / cuDNN / NCCL | AutoDL：驱动 595.71.05；`torch.version.cuda` 12.8；cuDNN pip 包 9.19.0.56（随 torch 2.11）。NCCL 待 nsys/多卡时再记。 |
| Python / PyTorch / Triton / Nsight Systems | AutoDL：Python 3.12.3（miniconda），PyTorch 2.11.0+cu128，Triton 3.6.0。不要 `uv run`。Nsight Systems CLI 2026.5.1；6 组 `nsys_profile` 报告见 `nsys_reports/`，csv 见 `nsys_stats/`，数字底表见 `notes/nsys_profile_tables.md`。(a)–(e) 已整理，统计边界限制见对应答案。 |
| 多卡互联 / 通信后端 | 本章目前单卡；未测。 |
| 随机种子 / 预热 / 测量次数 | `seed=42`。表 1 (b) 为 warmup 5、测量 10。6000 上 small/medium/large/xl 已跑 `--timing stages` 及三种 `--timing total`。10B warmup 首次 forward OOM。(c) 在 6000 上对 small `--mode train --timing total` 扫了 warmup 0/1/2/5。 |
| 计时范围 / 同步方式 / 时间单位 | `timeit.default_timer`，每阶段/每步前后 `torch.cuda.synchronize()`；毫秒；标准差 `statistics.pstdev`。`zero_grad`、初始化、数据生成不计时。 |
| 显存指标 / 峰值统计区间 / 单位 | 测量区间 `max_memory_allocated` / `max_memory_reserved`，MiB。 |
| 与讲义指定硬件或设置的差异 | 非 B200。本章计时均来自 AutoDL 6000。表 1 (b)(c) 计时用 `torch.optim.AdamW`；`nsys_profile` 用 `--optimizer-class cs336_basics.optimizer:AdamW`。 |

### 默认模型配置（讲义 Table 1）

| 模型 | d_model | d_ff | 层数 | 头数 |
| --- | --- | --- | --- | --- |
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 2560 | 10240 | 32 | 32 |
| 10B | 4608 | 12288 | 50 | 36 |

默认词表大小 10,000、batch size 4、context length 512；具体题目另有规定时以题目为准。单卡与多卡结果分开记录，指定 B200 的实验与其他硬件的实测结果明确区分。

第 2 章在引入 mixed precision 前以 FP32 参数和激活为基线。硬件差异记录仅用于解释结果，不表示课程允许以其他硬件替代指定 B200 或多卡实验。

## 2. Profiling and Benchmarking

### benchmarking_script — Benchmarking Script（4 分）

讲义第 3–4 页。

#### (a) 计时脚本

**题目**

编写脚本：根据超参数初始化模型、生成随机 batch；支持 w 次预热与 n 次测量，以及仅前向、前向加反向、前向加反向加优化器更新三种模式。使用适当的计时工具，并按题目要求在每一步后进行 GPU 同步。

**交付要求**：实现脚本。

**答案**

- 实现文件与入口：`cs336_systems/benchmark.py`；仓库根目录 `python -m cs336_systems.benchmark`（AutoDL 需 `PYTHONPATH` 含 `cs336-basics` 与仓库根，conda `base`，不要 `uv run`）。
- 三种模式的计时范围：`--mode forward` 只计前向；`--mode forward_backward` 为前向+loss+反向；`--mode train` 再加 `optimizer.step`。`--timing total` 同步后测整段；`--timing stages` 在 forward / loss / backward / optimizer 之间同步，**阶段之和不作端到端**。
- 运行配置与验证记录：默认 vocab 10000、batch 4、context 512、FP32 参数。AutoDL 已完成 small/medium/large/xl 的 train stages，以及这四档各自三种 `--timing total`。10B 同配置 OOM。(c) 已在 6000 上对 small train total 跑 warmup 0/1/2（warmup 5 用 (b) 的 60.583 ± 0.752 ms）。

#### (b) 各模型耗时

**题目**

测量 Table 1 各模型的前向、反向和优化器更新耗时。使用 5 次预热、10 次正式测量，报告平均值和标准差，讨论测量是否存在明显波动。

**交付要求**：计时结果及 1–2 句分析。

**答案**

主表「前向 / 反向 / 优化器」来自 `--mode train --timing stages`。三种脚本模式的端到端时间来自 `--timing total`（small/medium/large/xl 已齐）。完整步骤与「前向+反向」不要用阶段加总。

| 模型 | 前向均值 ± 标准差（ms） | 反向均值 ± 标准差（ms） | 优化器均值 ± 标准差（ms） | 完整步骤均值 ± 标准差（ms） | 状态 |
| --- | --- | --- | --- | --- | --- |
| small | 19.859 ± 0.611 | 35.691 ± 1.502 | 8.334 ± 0.317 | 60.583 ± 0.752 | 6000；stages + total train，2026-09-16 |
| medium | 47.538 ± 0.697 | 95.274 ± 0.096 | 24.113 ± 0.086 | 164.884 ± 0.500 | 6000；stages + total train，2026-09-16 |
| large | 111.164 ± 0.462 | 217.619 ± 0.192 | 54.947 ± 0.195 | 380.215 ± 0.087 | 6000；stages + total train，2026-09-16 |
| xl | 329.053 ± 0.082 | 594.658 ± 0.381 | 188.653 ± 0.243 | 1108.998 ± 0.243 | 6000；stages + total train，2026-09-16 |
| 10B | OOM | OOM | OOM | OOM | 6000，默认 batch 4 / ctx 512 / FP32 train，warmup 首次 forward |

三种脚本模式中的“前向 + 反向”合计时间另记于下表，不能与上表的“反向本身”混淆。

| 模型 | 前向 + 反向均值 ± 标准差（ms） | 状态 |
| --- | --- | --- |
| small | 55.050 ± 2.577 | 6000，`--mode forward_backward --timing total`，2026-09-16 |
| medium | 141.808 ± 0.239 | 6000，`--mode forward_backward --timing total`，2026-09-16 |
| large | 326.770 ± 0.078 | 6000，`--mode forward_backward --timing total`，2026-09-16 |
| xl | 922.157 ± 0.356 | 6000，`--mode forward_backward --timing total`，2026-09-16 |
| 10B | 未提供独立 forward_backward 结果 | 已知 train 在预热前向 OOM；不能代替此模式的独立运行记录 |

small 三种 `--timing total` 模式：`--mode forward` 18.884 ± 0.206 ms（peak allocated 3915.05 / reserved 4048.00 MiB）；`--mode forward_backward` 55.050 ± 2.577 ms（4157.55 / 4288.00）；`--mode train` 60.583 ± 0.752 ms（5144.79 / 5456.00）。`--mode forward` 的 total 略低于 stages 的 forward 列（19.859 ms），因 stages 夹在完整 train 步里且阶段间有同步。

medium 三种 `--timing total`：forward 46.573 ± 0.140 ms（10573.98 / 10872.00 MiB）；forward_backward 141.808 ± 0.239 ms（10817.29 / 11256.00）；train 164.884 ± 0.500 ms（14048.74 / 14530.00）。stages 的 forward 列为 47.538 ms。

large 三种 `--timing total`：forward 110.682 ± 0.057 ms（20508.86 / 20576.00 MiB）；forward_backward 326.770 ± 0.078 ms（20751.37 / 20976.00）；train 380.215 ± 0.087 ms（28147.39 / 29776.00）。stages 的 forward 列为 111.164 ms。

xl 三种 `--timing total`：forward 328.563 ± 0.024 ms（40695.58 / 41452.00 MiB）；forward_backward 922.157 ± 0.356 ms（41118.37 / 42512.00）；train 1108.998 ± 0.243 ms（67110.95 / 71306.00）。stages 的 forward 列为 329.053 ms。

6000 train-stages 峰值显存（测量区间）：small 5144.79 / 5456.00 MiB；medium 14048.74 / 14530.00 MiB；large 28147.39 / 29776.00 MiB；xl 67110.95 / 71306.00 MiB。参数量 small 128,625,408；medium 423,183,360；large 969,411,840；xl 3,406,809,600。10B 在 warmup 第一次 `forward` 的 attention softmax（`nn_utils.softmax` → `torch.exp`）处 `torch.OutOfMemoryError`：试图再分配 144.00 MiB；卡容量 94.97 GiB，进程已占用约 94.87 GiB，PyTorch allocated 约 93.71 GiB。优化器均为 `torch.optim.AdamW`。loss 阶段 ~0.1–0.15 ms，不单独填表。

计时边界与各阶段测量方法：`stages` 下 forward / loss / backward / optimizer 分别同步计时；`total` 下整段 `run_step` 同步计时。清梯度在测量循环内但排除在计时外。

分析：small/medium/large/xl 的前向、反向与优化器耗时均随模型规模增大，反向约为前向的 1.80–2.00 倍；分阶段测量的相对标准差最高约 4.21%（small 反向），medium/large/xl 各阶段均低于 1.5%，整体波动较小。10B 在预热前向即 OOM，无法报告该配置耗时；端到端 train 时间低于阶段之和，与阶段间额外同步和不同测量运行相符，不能用阶段加总替代端到端结果。

#### (c) 预热的影响

**题目**

重复实验，比较不预热、预热 1 次、预热 2 次与原设置的结果。结果如何变化？为什么少量预热后的结果仍可能不同？

**交付要求**：2–3 句分析。

**答案**

AutoDL RTX PRO 6000。small、`--mode train --timing total`、batch 4、context 512、FP32、测量 10。每种 warmup 单独进程。warmup=5 与 (b) 为同一次运行。

| 模型 / 模式 | 预热次数 | 测量次数 | 均值（ms） | 标准差（ms） |
| --- | --- | --- | --- | --- |
| small / train | 0 | 10 | 106.431 | 138.343 |
| small / train | 1 | 10 | 59.559 | 1.366 |
| small / train | 2 | 10 | 59.444 | 0.391 |
| small / train | 5 | 10 | 60.583 | 0.752 |

warmup=0 的 10 个样本：521.448, 61.165, 61.149, 61.038, 59.21, 59.213, 59.225, 59.665, 62.335, 59.865 ms（首步 521.448 ms，其后回到约 59–62 ms）。

分析：不预热时首步耗时 521.448 ms，将 10 步均值和标准差拉高到 106.431 ± 138.343 ms；可能包含 CUDA 库/内核首次使用、内存分配及 AdamW 状态的延迟初始化等一次性开销。预热 1、2 次后分别为 59.559 ± 1.366 ms 和 59.444 ± 0.391 ms，已接近预热 5 次的 60.583 ± 0.752 ms；不同进程中的缓存、GPU 时钟及系统负载仍可能造成小幅差异，因此预热次数增加不保证均值单调下降。现有预热对照仅覆盖 small，不能据此声称其余模型也已完成 (c)。

### nsys_profile — Nsight Systems Profiling（5 分）

讲义第 6 页。选择 Table 1 中两种模型大小，以及三种大于 128 的 2 的幂次 context length，最大者为显存能容纳的最长长度。对前向、反向和优化器更新进行 profiling，按讲义排除预热区间；以下各题按所选配置填写。

- 两种模型：medium、large（AutoDL RTX PRO 6000，batch 4，FP32，`--mode train --timing total --nvtx`）
- 各模型的三种长度：medium 为 512 / 1024 / 2048（原记录称 2048 为最长可容纳 2 的幂；本次未附 medium@4096 OOM 日志，仍需核实）；large 为 256 / 512 / 1024（原记录称 large@2048 OOM，故选择最大 1024；本次未附该 OOM 日志）
- Profile 文件与 NVTX 区间：`nsys_reports/{medium,large}_{len}.nsys-rep`；`nsys profile --capture-range nvtx --nvtx-capture measurement -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0`；脚本 NVTX 含 `warmup` / `measurement` / `forward` / `backward` / `optimizer` / `scaled_dot_product_attention` 等。分析用数字底表：`notes/nsys_profile_tables.md`。本次 profile 为 warmup 1、measurement 1；与 benchmarking 的 5/10 设置不同，没有重复测量标准差。

填写方式：对每个实际 profile 分别回答 (a)–(e)，在答案中标明模型和长度；可复制答案区或扩展表格，不能仅用一组结果代替全部配置。

#### (a) 前向总时间

**题目**：前向总耗时是多少？与 Python 标准库计时结果是否一致？

**交付要求**：1–2 句。

**答案**

以下为汇总表中的 **CPU NVTX forward 区间**及 Python 同步计时；CPU 区间内部未同步，不能视为 GPU 完整前向延迟。

| 模型 / 长度 | NVTX forward CPU（ms） | Python forward-total（ms） | Python stages-forward（ms） | 比较 |
| --- | ---: | ---: | ---: | --- |
| medium / 512 | 48.779 | 46.573 ± 0.140 | 47.538 ± 0.697 | CPU 数值分别高约 4.7% / 2.6%，接近但计时口径不同 |
| medium / 1024 | 35.437 | 未提供 | 未提供 | 缺同配置前向对照 |
| medium / 2048 | 101.614 | 未提供 | 未提供 | 1260.158 ms 是 train-total，不能用作前向对照 |
| large / 256 | 48.367 | 未提供 | 未提供 | 缺同配置前向对照 |
| large / 512 | 73.157 | 110.682 ± 0.057 | 111.164 ± 0.462 | CPU 数值低约 33.9%，不能认为 GPU 前向变快 |
| large / 1024 | 140.558 | 未提供 | 未提供 | 934.667 ms 是 train-total，不能用作前向对照 |

medium@512 的数值较接近，但 large@512 明显偏短，符合 CPU 提交结束后 GPU 仍在执行的异步现象；因此现有表不能为六组配置给出可靠的完整 GPU 前向时间。需根据 forward 范围内 CUDA API 的 correlation ID 追踪对应 GPU kernel，报告其时间跨度，并对缺失配置补充无 profiler 的同步 Python 计时。

#### (b) 最耗时的 kernel

**题目**：前向中累计 GPU 时间最多的 CUDA kernel 是什么？单次前向调用多少次？计入反向后，最耗时的 kernel 是否改变？

**交付要求**：1–2 句。

**答案**

下表忠实保留底表的 kernel 名称缩写；“前向窗口”可能截断 GPU 工作，“整步”包含 loss 和 optimizer，**不等于题目要求的前向+反向**。

| 模型 / 长度 | 前向窗口中排名第一的 kernel | 窗口累计 ms / 次数 | 整步排名第一的 kernel | 整步累计 ms / 次数 |
| --- | --- | ---: | --- | ---: |
| medium / 512 | cutlass `sgemm_256x128_…_tn` | 26.245 / 144 | 同一 tn GEMM | 26.245 / 144 |
| medium / 1024 | cutlass `sgemm_128x256_…_tn` | 15.698 / 45 | 同一 tn GEMM | 60.840 / 169 |
| medium / 2048 | cutlass `sgemm_128x256_…_tn` | 21.441 / 18 | elementwise Mul | 148.567 / 388 |
| large / 256 | cutlass `sgemm_128x256_…_tn` | 26.918 / 89 | cutlass `sgemm_256x128_…_nn` | 36.128 / 253 |
| large / 512 | cutlass `sgemm_128x256_…_tn` | 53.701 / 164 | 同一 tn GEMM | 83.545 / 253 |
| large / 1024 | cutlass `sgemm_128x256_…_tn` | 72.082 / 116 | 同一 tn GEMM | 160.240 / 253 |

六个前向窗口的首位均为 GEMM；整步统计中 medium@2048 变成逐元素乘法，large@256 变成另一种 GEMM，其余四组保持相同缩写的 GEMM。当前不能把这些窗口计数当作完整单次前向调用次数，也不能据整步排名断言前向+反向的排名；需补导出按 CUDA launch 归属的完整 forward 与 forward+backward kernel 汇总及完整 kernel 名称。

#### (c) 非矩阵乘法开销

**题目**：哪些非矩阵乘法 kernel 占据了不可忽略的前向 CUDA 时间？

**交付要求**：1–2 句。

**答案**

| 模型 / 长度 | 前向窗口非 gemm-like 占比 | 可以确认的非矩阵操作 |
| --- | ---: | --- |
| medium / 512 | 20.5% | 底表列出各占比 ≥1% 的 Mul / Add / Div、where、exp_kernel、reduce Max、copy |
| medium / 1024 | 45.9% | 缺逐 kernel 明细，不能确认具体操作的排名或占比 |
| medium / 2048 | 59.1% | 缺逐 kernel 明细，不能确认具体操作的排名或占比 |
| large / 256 | 13.1% | 缺逐 kernel 明细，不能确认具体操作的排名或占比 |
| large / 512 | 20.2% | 缺逐 kernel 明细，不能确认具体操作的排名或占比 |
| large / 1024 | 40.9% | 缺逐 kernel 明细，不能确认具体操作的排名或占比 |

medium@512 中逐元素运算、mask 选择、指数、归约与拷贝均占据不可忽略的 GPU 时间，这与未融合 softmax、归一化及张量处理产生多个 kernel 相符。其余五组仅能报告窗口内非 gemm-like 的合计，尚不能完成逐操作识别；长序列窗口的非 GEMM 占比更高，但必须修正窗口截断并核对分类后才能作为完整前向趋势。

#### (d) 完整训练步骤的耗时构成

**题目**：对包含前向、loss、反向和 AdamW 更新的完整训练步骤进行 profiling。与仅前向相比，矩阵乘法和其他 kernel 的耗时占比如何变化？

**交付要求**：1–2 句。

**答案**

| 模型 / 长度 | 前向窗口 GPU 合计（ms） | 前向 GEMM / 其他 | 完整训练步 GPU 合计（ms） | 整步 GEMM / 其他 | GEMM 占比变化（百分点） |
| --- | ---: | --- | ---: | --- | ---: |
| medium / 512 | 47.113 | 79.5% / 20.5% | 159.422 | 61.2% / 38.8% | −18.3 |
| medium / 1024 | 34.883 | 54.1% / 45.9% | 431.122 | 45.1% / 54.9% | −9.0 |
| medium / 2048 | 101.132 | 40.9% / 59.1% | 1245.825 | 34.4% / 65.6% | −6.5 |
| large / 256 | 46.392 | 86.9% / 13.1% | 196.265 | 61.7% / 38.3% | −25.2 |
| large / 512 | 71.894 | 79.8% / 20.2% | 365.618 | 60.8% / 39.2% | −19.0 |
| large / 1024 | 139.807 | 59.1% / 40.9% | 918.900 | 50.0% / 50.0% | −9.1 |

现有粗分类中，六组完整训练步的 GEMM 占比均低于前向窗口，其他 kernel 占比相应增加；这与反向中的逐元素/归约操作以及自实现 AdamW 的参数、矩估计更新引入额外内存访问相符。这里的 GEMM 仅按 `gemm` / `cutlass` / `sgemm` 名称粗分，且前向来自训练图并可能截断，所以尚不能作为严格的 inference 与 train 对比：应补充关闭 autograd 的 inference profile，并按 kernel 实际功能复核分类。

各比例分母是各自窗口的 kernel duration 之和，不是墙钟时间；带嵌套 NVTX 的 Time(%) 也不能作为训练阶段占比。此处 AdamW 为 `cs336_basics.optimizer:AdamW`，不可与 benchmarking 中 `torch.optim.AdamW` 的优化器耗时直接比较。

#### (e) Softmax 与矩阵乘法

**题目**：比较前向 Attention 内 softmax 与矩阵乘法的耗时；耗时差异与 FLOPs 差异如何对应？

**交付要求**：1–2 句。

**答案**

下表统计所有层的 **CPU NVTX 时间**（medium 24 层、large 36 层），未把默认只取第一次 attention 的过滤表当作全层总和。

| 模型 / 长度 | QKᵀ CPU（ms） | PV CPU（ms） | 两个 matmul 合计（ms） | softmax CPU（ms） | softmax / 两个 matmul |
| --- | ---: | ---: | ---: | ---: | ---: |
| medium / 512 | 3.220 | 2.475 | 5.695 | 4.584 | 0.805 |
| medium / 1024 | 2.243 | 1.841 | 4.084 | 3.774 | 0.924 |
| medium / 2048 | 1.269 | 1.756 | 3.025 | 45.263 | 14.963 |
| large / 256 | 2.069 | 2.805 | 4.874 | 3.435 | 0.705 |
| large / 512 | 7.372 | 3.703 | 11.075 | 10.152 | 0.917 |
| large / 1024 | 4.053 | 2.980 | 7.033 | 9.259 | 1.317 |

每层两个 attention 矩阵乘法合计约为 $4BHL^2d_h$ FLOPs，而 softmax 为 $O(BHL^2)$；这两档模型的 $d_h=64$，因此 matmul FLOPs 显著更多，但 softmax 的多次读写、归约、指数和 kernel 启动会使其耗时不按 FLOPs 比例缩小。表中 softmax 的 CPU 区间为 matmul 合计的 0.705–14.963 倍，不过异步提交及排队会扭曲该比值，尤其不能将 medium@2048 的 14.963 倍解释为 GPU softmax 慢 15 倍；实际 GPU 比值仍须按 CUDA API 关联补导出。

softmax 应覆盖其 max、减法、exp、sum、除法对应的 kernel；不能把 loss 的 `cunn_SoftMax*` 自动归为 attention softmax。

### mixed_precision_accumulation — Mixed-Precision Accumulation（1 分）

讲义第 7–8 页。

**题目**

运行讲义给出的四组累加实验，均为反复累加 0.01，共 1000 次。比较不同输入与累加器精度、显式类型转换下的结果，并评论精度。具体表达式以讲义为准。

**交付要求**：2–3 句分析。

**答案**

精确数学结果为 $1000\times0.01=10$。当前环境未安装 PyTorch，下列数值用 NumPy 按相同 FP16/FP32 格式逐步舍入复核，属于原 PyTorch 程序的预期结果，**不是已执行原程序的日志**。

| 讲义实验 | 同精度复核值（待 PyTorch 原代码确认） | 有符号误差：结果 − 10 | 绝对相对误差 |
| --- | ---: | ---: | ---: |
| 1：FP32 累加器 / FP32 输入 | 10.0001335144 | +0.0001335144 | 0.001335% |
| 2：FP16 累加器 / FP16 输入 | 9.9531250000 | −0.0468750000 | 0.468750% |
| 3：FP32 累加器 / FP16 输入 | 10.0021362305 | +0.0021362305 | 0.021362% |
| 4：FP16 输入显式转为 FP32 后累加 | 10.0021362305 | +0.0021362305 | 0.021362% |

FP32 累加最接近 10，而 FP16 累加在每次写回累加器时舍入，导致更大的累计误差。第 3、4 组均使用 FP32 累加，因此结果一致且优于全 FP16；它们仍保留了输入在 FP16 中由 0.01 舍入为 0.01000213623046875 的误差，后续显式转为 FP32 无法恢复已丢失的精度。

### benchmarking_mixed_precision — Benchmarking Mixed Precision（2 分）

讲义第 8–9 页。

#### (a) ToyModel 的数据类型

**题目**：使用第 8 页的 ToyModel，参数初始为 FP32，在 GPU 上使用 FP16 autocast。分别报告参数、fc1 输出、LayerNorm 输出、logits、loss 和参数梯度的 dtype。

**交付要求**：列出每项数据类型。

**答案**

按 CUDA FP16 autocast 规则回答，假定 loss 是在 autocast 范围内调用的 `torch.nn.functional.cross_entropy`，参数保持原始 FP32；这是 dtype 推导，未在本次环境运行 GPU ToyModel。

| 对象 | dtype |
| --- | --- |
| autocast 内的模型参数 | `torch.float32` |
| fc1 输出 | `torch.float16` |
| LayerNorm 输出 | `torch.float32` |
| logits（fc2 输出） | `torch.float16` |
| loss（上述 cross_entropy） | `torch.float32` |
| 参数梯度（FP32 参数的 `.grad`） | `torch.float32` |

autocast 按操作选择计算精度，不会永久把参数改成 FP16；`linear` 使用低精度，`layer_norm` 和 `cross_entropy` 使用 FP32。题目未具体指定 loss 算子，因此 loss dtype 的结论依赖上述假设。依据：[PyTorch 2.11 AMP 的 CUDA 操作规则](https://docs.pytorch.org/docs/2.11/amp.html#cuda-op-specific-behavior)。

#### (b) 归一化与精度

**题目**：LayerNorm 的哪些部分对混合精度敏感？如果使用 BF16 而不是 FP16，是否仍需对 LayerNorm 特殊处理？为什么？

**交付要求**：2–3 句。

**答案**

LayerNorm 的均值、方差归约及去均值过程容易受到累加舍入和相近数相减的影响，平方与逆平方根又会放大动态范围不足或小方差的数值问题，因此这些计算宜使用 FP32。BF16 的指数范围与 FP32 相同，可缓解 FP16 的上溢/下溢，但其尾数精度更低，不能消除统计量的舍入误差，所以仍应保留高精度归约，而不是把整个 LayerNorm 直接改成纯 BF16。

#### (c) BF16 性能比较

**题目**：给计时脚本增加可选 BF16 混合精度模式。对 Table 1 各模型比较全精度与混合精度的前向、反向耗时，讨论随模型大小变化的趋势。

**交付要求**：计时结果及 2–3 句分析。

**答案**

可使用 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` 包住 forward 和 loss，保持模型参数 FP32，backward 在 autocast 上下文之外执行；本次未附脚本，无法验证该选项的实际实现。

| 模型 | FP32 前向（ms） | BF16 mixed 前向（ms） | FP32 反向（ms） | BF16 mixed 反向（ms） | 状态 |
| --- | --- | --- | --- | --- | --- |
| small | 19.859 ± 0.611 | 待补测 | 35.691 ± 1.502 | 待补测 | FP32 取自前述 stages |
| medium | 47.538 ± 0.697 | 待补测 | 95.274 ± 0.096 | 待补测 | 同上 |
| large | 111.164 ± 0.462 | 待补测 | 217.619 ± 0.192 | 待补测 | 同上 |
| xl | 329.053 ± 0.082 | 待补测 | 594.658 ± 0.381 | 待补测 | 同上 |
| 10B | OOM（train 配置） | 待补测 | OOM（未进入 backward） | 待补测 | 不代表 BF16 也 OOM |

已有材料没有 BF16 耗时，无法计算实测加速比或声称加速随模型增大。理论上低精度矩阵乘法吞吐提升及激活流量减少可能带来收益，但类型转换、FP32 操作与 kernel 启动会影响实际加速；须在相同 batch/context、warmup/measurement、优化器和同步边界下补测。

### memory_profiling — Memory Profiling（4 分）

讲义第 9–10 页。使用 xl 模型、context length 128 和 2048。

#### (a) 显存时间线

**题目**：为脚本增加 memory profiling 选项，对比仅前向与完整训练步骤的 Active memory timeline。能否根据峰值辨认执行阶段？

**交付要求**：两张时间线截图（仅前向、完整步骤）和 2–3 句分析，明确截图的长度与精度等配置。

**答案**

- 实现状态：本次未附脚本，无法核验 memory profiling 选项；讲义要求在预热后调用 `_record_memory_history`，测量后 `_dump_snapshot`，最后停止记录。
- 仅前向截图：待提供 xl 的 memory snapshot 和 Active memory timeline；应明确 context 128 或 2048、FP32/BF16，以及是否用 `inference_mode()` 或 `no_grad()`。
- 完整步骤截图：待提供相同模型和明确配置的 forward + loss + backward + optimizer 时间线。

理论上 inference 无需为反向保留各层激活，而训练前向会积累保存张量，反向逐步释放这些张量并产生参数梯度，优化器阶段还可能出现状态或临时张量分配峰值。实际峰值所处阶段必须结合时间线与 NVTX 判断，不能仅凭这个预期给现有记录标注阶段；预热后 AdamW 状态通常已建立，不能把每个 optimizer 峰值都解释成首次状态初始化。

#### (b) 峰值显存

**题目**：两个长度在仅前向与完整训练步骤中的峰值显存分别是多少？

**交付要求**：每种长度两个数值的表格。

**答案**

| Context length | 仅前向峰值（GiB） | 完整步骤峰值（GiB） |
| --- | --- | --- |
| 128 | 待补测 | 待补测 |
| 2048 | 待补测 | 待补测 |

现有 xl@512 的训练图 forward / train 记录不属于本题指定长度，也不等同于 inference，不能填入上表。建议以测量前重置峰值后的 `max_memory_allocated()/1024**3` 为数值口径，同时单列 reserved，勿混用两者。

#### (c) 混合精度显存

**题目**：测量混合精度下 xl 的仅前向与完整步骤峰值显存。混合精度是否显著影响显存占用？

**交付要求**：2–3 句，包含结果。

**答案**

| 长度 | 精度 | 仅前向峰值（GiB） | 完整步骤峰值（GiB） |
| --- | --- | --- | --- |
| 128 | FP32 | 待补测 | 待补测 |
| 128 | BF16 mixed | 待补测 | 待补测 |
| 2048 | FP32 | 待补测 | 待补测 |
| 2048 | BF16 mixed | 待补测 | 待补测 |

目前缺上述配置的结果，无法判断实测降幅是否显著。BF16 autocast 可以减少部分激活及中间结果大小，但参数、FP32 参数梯度和通常的 FP32 AdamW 状态仍占显存，且可能存在低精度权重缓存，所以总峰值不会必然减半；训练与 inference 的降幅也可能不同。

#### (d) 残差流激活大小

**题目**：在 xl 参考超参数下，单精度的一个 Transformer residual stream 激活张量有多大？以 MiB 表示（字节数除以 1024²），写明采用的配置。

**交付要求**：1–2 句及推导。

**答案**

一个残差流激活的形状为 $(B,L,d_{model})$，xl 使用 $B=4$、$d_{model}=2560$、FP32 每元素 4 字节，因此大小为

$$
M=\frac{4\times L\times2560\times4}{1024^2}=0.0390625L\ \mathrm{MiB}.
$$

按 Table 1 默认 $L=512$，结果为 **20 MiB**；本题显存实验的 $L=128$ 与 $L=2048$ 则分别为 **5 MiB** 和 **80 MiB**，这些均为单个张量，不是所有层保存张量总和。

#### (e) 最大分配的来源

**题目**：在 xl 前向的 Active Memory Timeline 中降低 Detail，观察最大分配的大小，并根据 stack trace 确定其来源。

**交付要求**：1–2 句。

**答案**

尚未提供 xl 的 memory snapshot 或分配调用栈，因此无法确定“实际最大分配”的大小及来源。作为核验线索，FP32 attention score/probability 张量 `(4,32,L,L)` 在 L=128 时为 8 MiB、L=2048 时为 2048 MiB，而单个 FFN 权重 `(2560,10240)` 为 100 MiB；这些只是候选张量的理论大小，必须由实际 stack trace 确认，不能代替观测结论。

#### (f) Block 保存张量与梯度显存

**题目**：使用 Nsight 内存分析与 NVTX 标签，测量单个 TransformerBlock 为反向保存的张量占用；列出贡献最大的五个操作及占比。结合前向分配和反向显存变化，计算该 block 产生的梯度张量占用，并与预期比较。

**交付要求**：Nsight 截图及 1–2 段分析。

**答案**

缺少按 TransformerBlock 标注的内存分配事件、保存张量记录及 Nsight 截图，因此目前不能给出“最大的五个操作”及百分比；不能直接把 kernel 耗时排名当成保存内存排名。

| 排名 | 保存张量对应操作 | 保存内存（MiB） | 占总保存张量比例 |
| --- | --- | --- | --- |
| 1 | 待补测 | 待补测 | 待补测 |
| 2 | 待补测 | 待补测 | 待补测 |
| 3 | 待补测 | 待补测 | 待补测 |
| 4 | 待补测 | 待补测 | 待补测 |
| 5 | 待补测 | 待补测 | 待补测 |

- 保存张量总量与截图：待提供；统计时应按底层 storage 去重，避免把共享视图或已有参数算成新增保存激活。
- 理论参数梯度：假定标准无 bias 多头注意力、三矩阵 SwiGLU 和两个 RMSNorm，单个 xl block 参数数目为 $P_{block}=4d^2+3d\,d_{ff}+2d=104862720$，若 `.grad` 为 FP32，持久参数梯度为 $4P_{block}/1024^2=400.01953125$ MiB；架构须与实际源码核对。

若一个 block 反向前后临时张量均已释放，且其保存激活净释放 $S$ MiB、边界处激活梯度等其他存活内存净变化为 $\Delta M_{other}$，则新增参数梯度为 $G=(M_{after}-M_{before})+S-\Delta M_{other}$。对干净边界、其他内存净变化可忽略且参数梯度尚未分配的情况，该值应接近上述 400.02 MiB；若 `.grad` 已存在并复用、使用 `set_to_none=False`、有共享参数或仍存活的临时张量，就不能直接套用这个简化式，也不能把参数梯度总量当作所有反向中间梯度总量。

## 3. Single-GPU Memory

### gradient_checkpointing — Memory-Optimal Gradient Checkpointing（4 分）

讲义第 15 页。

#### (a) 允许嵌套的检查点策略

**题目**

考虑由 N 个相同 block 顺序堆叠的 Transformer，不使用 checkpoint 时峰值激活内存为 O(N)。允许对任意前向部分使用 checkpoint，也允许嵌套。忽略计算代价时，什么策略最小化峰值激活内存？描述安排，给出关于 N 的内存与计算渐近复杂度。假定单 block 的保存张量主导每个 checkpoint 的记账开销。

**交付要求**：3–5 句策略说明、渐近峰值内存，以及自己编写的简短代码草图；同时回答题目要求的计算复杂度。

**答案**

- 策略说明：【待填写】
- 峰值激活内存复杂度：【待填写】
- 计算复杂度：【待填写】
- 自己编写的简短代码草图：【待在报告中直接填写；仅列文件路径不能替代本项交付】

#### (b) 不允许嵌套的实验

**题目**：使用 xl、batch size 4、序列长度 2048。只允许一层重计算、不允许嵌套 checkpoint 时，什么安排最能降低峰值内存？用 profiling 验证，并与相邻更小、更大的 checkpoint block size 比较。

**交付要求**：3–5 句推理及实测峰值。

**答案**

| 设置 | Checkpoint block size | 峰值显存（GiB） |
| --- | --- | --- |
| 相邻更小 | 待填写 | 待填写 |
| 所选设置 | 待填写 | 待填写 |
| 相邻更大 | 待填写 | 待填写 |

推理与结论：【待填写】

## 4. GPU Kernels

### pytorch_attention — PyTorch Attention Benchmarking（2 分）

讲义第 16 页。

#### (a) 基准与显存分析

**题目**

固定 batch size 8，移除多头维度。遍历 d ∈ {16, 32, 64, 128} 与序列长度 ∈ {256, 1024, 4096, 8192, 16384} 的笛卡尔积，随机生成 Q/K/V；先预热，在每次前向 / 反向后进行 GPU 同步，计时 100 次前向及 100 次反向，记录反向开始前的显存占用。报告耗时或 OOM，并回答从哪些规模开始 OOM；选择一个最小级别的 OOM 配置做显存核算，讨论为反向保存的内存如何随长度变化，以及怎样消除这部分成本。

**交付要求**：完整结果表、显存计算及 1–2 段分析。

**答案**

下表扩展为全部 20 组配置。

| d | 长度 | 前向（ms） | 反向（ms） | 反向前显存（MiB） | 状态 |
| --- | --- | --- | --- | --- | --- |
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待运行 |

- OOM 配置及显存核算：【待填写】
- 随长度变化的分析：【待填写】
- 关于消除内存成本的回答：【待填写】

### torch_compile — Torch Compile（2 分）

讲义第 16–17 页。

#### (a) 编译 Attention

**题目**：在上一题相同配置下，比较原始 PyTorch Attention 与 torch.compile 版本的前向和反向时间。

**交付要求**：覆盖上一题配置的对比表。

**答案**

| d | 长度 | 原始前向（ms） | 编译前向（ms） | 原始反向（ms） | 编译反向（ms） | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待运行 |

#### (b) 编译完整 Transformer

**题目**：在端到端脚本中编译整个 Transformer。前向和包含前向、反向及优化器更新的训练步骤性能如何变化？

**交付要求**：原始与编译模型的对比表。

**答案**

| 模型 / 长度 | 版本 | 前向（ms） | 前向 + 反向（ms） | 完整步骤（ms） |
| --- | --- | --- | --- | --- |
| 待填写 | 原始 | 待填写 | 待填写 | 待填写 |
| 待填写 | 编译 | 待填写 | 待填写 | 待填写 |

实验设置与观察：【待填写】

### flash_forward — FlashAttention-2 Forward Pass（15 分）

讲义第 26–27 页。

#### (a) PyTorch 前向

**题目**：使用纯 PyTorch autograd.Function 实现 FlashAttention-2 前向，输入 Q/K/V 和可暂时忽略的 is_causal 标志；计算 O 与 logsumexp L，保存 L/Q/K/V/O 供反向使用并返回 O。tile 至少为 16 × 16，测试维度为不小于 16 的 2 的幂。具体接口与公式见讲义。

**交付要求**：实现、接入 `adapters.get_flashattention_autograd_function_pytorch`，并通过 `test_flash_forward_pass_pytorch`。本阶段允许反向暂未实现，具体接口及占位要求见讲义第 26 页。

**答案**

- 实现文件与入口：【待填写】
- Adapter：【待填写】
- 测试结果与日志：【待填写】

#### (b) Triton 前向

**题目**：按 Algorithm 1 实现 Triton FlashAttention-2 前向，并通过另一个 autograd.Function 调用。遵守第 26–27 页的 launch grid、循环、指针、精度及接口要求。

题目列出的具体约束与精度指导（不含实现）：

- Launch grid 为 (T_q, batch_size)，每个 program instance 对应单个 batch 的一个 query tile。
- Kernel 只包含遍历 key tiles 的一个循环，并在循环末尾推进 block pointers。
- 使用讲义给定的 kernel 参数声明及 Q block pointer 约定。
- 片上 O_i、l、m 缓冲使用 FP32；矩阵乘法的累加方式遵循讲义说明。
- P̃ 与 V 相乘前的数据类型转换，以及 O 写回前的数据类型转换，遵循第 27 页要求。

**交付要求**：实现并通过 `test_flash_forward_pass_triton`。

**接口名称核对**：PDF 第 27 页写作 `adapters.get_flash_autograd_function_triton`，但当前仓库 `tests/adapters.py` 实际定义为 `get_flashattention_autograd_function_triton`。接入本地测试时核对后者；这是 PDF 与代码的名称差异，不是两道不同题。

**答案**

- 实现文件与入口：【待填写】
- Adapter：【待填写】
- 测试结果与日志：【待填写】

#### (c) 因果遮罩

**题目**：在 autograd.Function 的最后一个参数位置增加可选布尔标志 is_causal，默认 False。Triton 对应参数必须使用 `tl.constexpr` 类型标注。按 query/key 索引比较形成 B_q × B_k 遮罩；题目指定对被遮挡的 attention score 加上 −1e6。将该标志保存在 context 的 is_causal 属性中供反向使用。

**交付要求**：支持 causal masking 的前向实现，并保持已有测试兼容。

**答案**

- 实现位置：【待填写】
- 遮罩开启 / 关闭的验证结果：【待填写】

### flash_backward — FlashAttention-2 Backward Pass（5 分）

讲义第 28 页。

**题目**：使用 PyTorch 与 torch.compile（本题不要求 Triton）实现反向。输入 Q/K/V/O/dO/L，输出 dQ/dK/dV；按讲义要求计算并使用 D，相关计算见式 (13)–(19)。

**交付要求**：实现并通过 `test_flash_backward`。

**答案**

- 实现文件与入口：【待填写】
- 测试结果与日志：【待填写】

### flash_benchmarking — FlashAttention-2 Benchmarking（5 分）

讲义第 28 页。

#### (a) 性能比较

**题目**

使用 triton.testing.do_bench 比较自己的 FlashAttention-2 与普通 PyTorch Attention，分别测前向、反向及前向加反向。输入预先生成；指定单张 B200、batch size 1、causal masking。扫描序列长度 128 至 65536 的 2 的幂、维度 16 至 128 的 2 的幂，以及 BF16 / FP32 的笛卡尔积。

**交付要求**：上述配置的完整延迟表；使用其他硬件时明确标注实测设备和差异。

**答案**

| 实际 GPU | 长度 | 维度 | dtype | 实现 | 前向（ms） | 反向（ms） | 前向 + 反向（ms） | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待填写 | 待填写 | 待填写 | 待填写 | PyTorch | 待填写 | 待填写 | 待填写 | 待运行 |
| 待填写 | 待填写 | 待填写 | 待填写 | FlashAttention-2 | 待填写 | 待填写 | 待填写 | 待运行 |

- 预热 / 测量配置 / tile 设置：【待填写】
- 完整原始结果文件：【待填写】

### 4.2.3 可选：Triton backward pass

**题目**：讲义第 28–29 页提供可选 Triton 反向扩展（Algorithm 2），未单列分值。

本区不替代 `flash_backward` 必做题；第 4.2.1 节 Weighted Sum 是讲解示例，没有独立 Problem 或 writeup 交付项。

**答案**

- 是否完成：【待填写】
- 实现位置与验证结果：【待填写】
- 性能记录（如有）：【待填写】

## 5. Distributed Data Parallel Training

### distributed_communication_single_node — Distributed Communication (Single Node)（5 分）

讲义第 32 页。

**题目**：编写单节点多进程 all-reduce 基准脚本，使用 float32 张量，数据量为 1MB、10MB、100MB、1GB，GPU / 进程数为 2、4、6。最多需要 6 张 GPU，每次 benchmark 应少于 5 分钟；遵循讲义的预热、同步与跨 rank 测量规范。

**交付要求**：不同设置的图和/或表，以及 2–3 句分析。

**答案**

- 脚本位置 / 后端 / 各 rank 统计方式：【待填写】
- 实际硬件与未完成的配置：【待填写】

| 数据量 | 2 GPU / 进程（ms） | 4 GPU / 进程（ms） | 6 GPU / 进程（ms） |
| --- | --- | --- | --- |
| 1MB | 待填写 | 待填写 | 待填写 |
| 10MB | 待填写 | 待填写 | 待填写 |
| 100MB | 待填写 | 待填写 | 待填写 |
| 1GB | 待填写 | 待填写 | 待填写 |

图（如有）：【待插入】

分析：【待填写】

### naive_ddp — Naïve DDP（5 分）

讲义第 33 页。

**题目**：实现基础 DDP，在反向完成后对各个参数梯度分别 all-reduce。

**交付要求**：接入 `adapters.get_ddp` 及按需使用的 `adapters.ddp_on_after_backward`，通过 `tests/test_ddp.py`。

**答案**

- 实现文件与入口：【待填写】
- Adapter：【待填写】
- 测试结果与日志：【待填写】

### naive_ddp_benchmarking — Naïve DDP Benchmarking（3 分）

讲义第 33 页。

**题目**：对基础 DDP 进行基准测试，测量每个完整训练步骤时间、梯度通信时间及其占比。指定单节点、2 张 GPU、xl 模型。

**交付要求**：实验设置说明，以及训练迭代与梯度通信时间。

**答案**

- 实验设置 / 计时范围：【待填写】

| 配置 | 每步时间（ms） | 梯度通信时间（ms） | 通信占比 |
| --- | --- | --- | --- |
| 1 node × 2 GPUs，xl | 待填写 | 待填写 | 待填写 |

### minimal_ddp_flat_benchmarking — Minimal DDP with Flat Gradients Benchmarking（2 分）

讲义第 34 页。

**题目**：修改基础 DDP，对所有参数梯度展平后的单个张量进行一次 all-reduce。在同一设置（单节点、2 GPU、xl）下与逐参数通信比较。

**交付要求**：每步训练时间、梯度通信时间，及 1–2 句比较。

**答案**

| 实现 | 每步时间（ms） | 梯度通信时间（ms） |
| --- | --- | --- |
| 逐参数 all-reduce | 待填写 | 待填写 |
| 单个展平张量 all-reduce | 待填写 | 待填写 |

比较：【待填写】

### ddp_overlap_individual_parameters — DDP with Overlapping Individual Parameters（5 分）

讲义第 35–36 页。

**题目**：实现包装 nn.Module 的 DDP 容器，保证初始参数一致，支持梯度平均，并使逐参数梯度通信与反向计算重叠。接口及同步要求见讲义。

讲义建议的公开接口为初始化、forward 和 `finish_gradient_synchronization`；forward 应转发原模型的输入参数，梯度同步完成的等待发生在优化器更新之前。

**交付要求**：实现、接入 DDP adapters，并通过 `tests/test_ddp.py`。讲义建议重复测试（例如 5 次）确认稳定性。

**答案**

- 实现文件 / 接口 / Adapter：【待填写】
- 测试次数、结果与日志：【待填写】

### ddp_overlap_individual_parameters_benchmarking — DDP Overlapping Individual Parameters Benchmarking（1 分）

讲义第 36 页。

#### (a) 性能比较

**题目**：在单节点、2 GPU、xl 下，比较通信与反向重叠的实现、逐参数基础 DDP、单次展平 all-reduce DDP。

**交付要求**：每步耗时与 1–2 句比较。

**答案**

| 实现 | 每步时间（ms） |
| --- | --- |
| 基础 DDP：逐参数通信 | 待填写 |
| 基础 DDP：展平通信 | 待填写 |
| 逐参数通信与反向重叠 | 待填写 |

比较：【待填写】

#### (b) 时间线证据

**题目**：使用 Nsight 比较初始 DDP 与重叠实现的时间线，展示通信与反向计算是否重叠。配置仍为单节点、2 GPU、xl。

**交付要求**：两张截图，分别对应初始实现和重叠实现。

**答案**

- 初始 DDP 截图及图注：【待插入】
- 重叠 DDP 截图及图注：【待插入】

## 6. Optimizer State Sharding

### optimizer_state_sharding — Optimizer State Sharding（15 分）

讲义第 37 页。

**题目**：实现优化器状态分片容器，封装任意输入 PyTorch Optimizer，在每次更新后同步更新过的参数。支持参数集合与 parameter groups、优化器参数转发、step 的 closure 与其他参数，以及训练期间添加 parameter group。构造与接口要求见讲义。

第 37 页还明确要求：初始化时调用 Optimizer 父类构造器；`add_param_group` 不仅会在训练中调用，也会由父类构造器调用，两种场景都需支持。

**交付要求**：实现并接入 `adapters.get_sharded_optimizer`，通过 `tests/test_sharded_optimizer.py`；讲义建议重复测试（例如 5 次）。

**答案**

- 实现文件与入口：【待填写】
- Adapter：【待填写】
- 测试次数、结果与日志：【待填写】

### optimizer_state_sharding_accounting — Optimizer State Sharding Accounting（5 分）

讲义第 38 页。标准配置：单节点、2 GPU、xl。

#### (a) 显存分析

**题目**：比较启用 / 不启用优化器状态分片，在模型初始化后、优化器更新前、优化器更新后的峰值显存。结果是否符合预期？拆分参数、优化器状态等组成。

**交付要求**：峰值与组成记录，以及 2–3 句说明。

**答案**

| 设置 | 初始化后峰值（GiB） | 更新前峰值（GiB） | 更新后峰值（GiB） |
| --- | --- | --- | --- |
| 不分片 | 待填写 | 待填写 | 待填写 |
| 优化器状态分片 | 待填写 | 待填写 | 待填写 |

| 设置 / 阶段 | 参数 | 梯度 | 优化器状态 | 激活 / 其他 | 单位 |
| --- | --- | --- | --- | --- | --- |
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

峰值统计区间与分析：【待填写】

#### (b) 训练速度

**题目**：优化器状态分片如何影响训练速度？测量标准配置下启用 / 不启用分片的每次迭代时间。

**交付要求**：计时与 2–3 句分析。

**答案**

| 设置 | 每步时间（ms） |
| --- | --- |
| 不分片 | 待填写 |
| 优化器状态分片 | 待填写 |

分析：【待填写】

#### (c) 与 ZeRO Stage 1 的比较

**题目**：本作业的优化器状态分片方法与讲义参考文献 [5] 中 ZeRO Stage 1（ZeRO-DP P_os）有何不同？重点比较内存与通信量。

**交付要求**：2–3 句。

**答案**

【待填写；注明参考文献中的对应位置】

## 7. Fully-Sharded Data Parallel

### fsdp — Fully-Sharded Data Parallel（15 分）

讲义第 38–39 页。

**题目**

实现包装 nn.Module 的 FSDP 容器，处理其中 Linear / Embedding 的权重分片与通信。满足前向和反向所需的权重 all-gather、梯度 reduce-scatter、权重使用后释放及最终同步要求。前向预取时机须符合讲义约束：目标层前面第二层完成前向后才开始收集。指定 compute_dtype 时，在通信及计算前转换权重，同时保留 FP32 主权重与优化器更新。具体接口以讲义为准。

讲义第 38 页还要求将不值得分片的小层标为不分片（本架构主要是归一化层）。第 39 页建议初始化、forward 和 `finish_gradient_synchronization` 接口；forward 转发原模型的输入参数。

**交付要求**：与 Assignment 1 AdamW 兼容的实现，接入 `adapters.get_fsdp`，通过 `tests/test_fsdp.py`。建议重复测试（例如 5 次）排查竞争条件。

本地测试补充说明（不是新增 PDF 题目）：当前 `tests/adapters.py` 还定义了 `fsdp_on_after_backward` 与 `fsdp_gather_full_params`，测试接入时一并核对这些接口的文档和调用处。

**答案**

- 实现文件与入口：【待填写】
- Adapter：【待填写】
- 测试次数、结果与日志：【待填写】

### fsdp_accounting — FSDP Accounting（5 分）

讲义第 39 页。

#### (a) 预期显存节省

**题目**：结合第 6 章分析，实现 FSDP 后预期峰值显存能节省多少？计算中可忽略为 all-gather 预分配的缓冲区。

**交付要求**：2–3 句。

**答案**

【待填写推导、对比基线和结果】

#### (b) 权重通信能否及时完成

**题目**：在两张 GPU 上 profile xl 模型，关注权重 all-gather。通信能否及时完成以供前向使用？

**交付要求**：计时、2–3 句分析，以及支持结论的 Nsight 截图。

**答案**

- 测量配置及时间：【待填写】
- Nsight 截图与图注：【待插入】
- 分析：【待填写】

## 8. Analyzing Parallelism Strategies

### 共同设定（题目给定条件）

使用讲义第 40–46 页的理想化模型与公式，不以本地硬件参数替代符号：

- N：设备数；S：通信张量字节数；W：每设备出口带宽（bytes/s）。
- C：每设备计算能力（FLOP/s）。
- FFN 输入形状为 (B, D)；W₁、W₂ 为 (D, D_FF)，W₃ 为 (D_FF, D)。
- FFN 前向 / 非分片反向由式 (20)–(30) 给定；各并行策略的具体操作按对应章节。
- DP / FSDP / TP 计算题沿用 FP16 权重与激活（每元素 2 字节），忽略非矩阵乘法运算。
- 计算与通信可重叠的假设、二维通信能否重叠的假设，以每个子题为准。

### alternate_ring_all_reduce — Alternate Ring All-Reduce（1 分）

讲义第 41 页。

**题目**：考虑讲义给出的替代 ring all-reduce 算法：N−1 轮中各设备沿环传递相应完整输入张量并累加，而非先 reduce-scatter 再 all-gather。在每个输入大小 S、每设备出口带宽 W 的条件下，该算法耗时多久？传递索引和具体算法以第 41 页为准。

**交付要求**：用 S、N、W 表达的时间及一句理由。

**答案**

- 时间表达式：【待填写】
- 理由：【待填写】

### data_parallel_calcs — Data Parallel Calculations（3 分）

讲义第 42 页。按第 8.2 节的 DP 设置分析，通信超过计算时视为通信瓶颈。

#### (a) 反向计算量

**题目**：N_DP 路数据并行时，反向需要多少 FLOPs？忽略非矩阵乘法；矩阵 (A, B) 与 (B, C) 相乘按 2ABC FLOPs 计。

**交付要求**：含 B、D、D_FF、N_DP 的表达式及一句理由。

**答案**

- 表达式：【待填写】
- 理由：【待填写】

#### (b) 反向通信时间

**题目**：N_DP 路数据并行的反向通信耗时是多少？

**交付要求**：使用 B、D、D_FF、N_DP、W 中所需变量的表达式及一句理由。

**答案**

- 表达式：【待填写】
- 理由：【待填写】

#### (c) 可扩展设备数

**题目**：固定其他参数，N_DP 最大能增加到什么程度而不成为通信瓶颈？

**交付要求**：一侧为 N_DP 的不等式，另一侧使用 B、D、D_FF、C、W 中所需变量，附一句理由。

**答案**

- 不等式：【待填写】
- 理由：【待填写】

### fsdp_calcs — Fully Sharded Data Parallel Calculations（3 分）

讲义第 43–44 页。沿用 DP 的假设，FSDP 通信操作按式 (34)–(44)。

#### (a) 前向与反向计算量

**题目**：N_FSDP 路 FSDP 的反向和前向分别需要多少 FLOPs？

**交付要求**：两个含 B、D、D_FF、N_FSDP 的表达式，各附一句理由。

**答案**

- 反向表达式与理由：【待填写】
- 前向表达式与理由：【待填写】

#### (b) 前向与反向通信时间

**题目**：N_FSDP 路 FSDP 的反向和前向通信分别耗时多久？

**交付要求**：两个表达式，使用 B、D、D_FF、N_FSDP、W 中所需变量，各附一句理由。

**答案**

- 反向表达式与理由：【待填写】
- 前向表达式与理由：【待填写】

#### (c) 可扩展设备数

**题目**：固定其他参数，反向与前向分别允许 N_FSDP 增加到什么程度而不成为通信瓶颈？

**交付要求**：两个一侧为 N_FSDP 的不等式，使用 B、D、D_FF、C、W 中所需变量，各附一句理由。

**答案**

- 反向约束与理由：【待填写】
- 前向约束与理由：【待填写】

### tp_calcs — Tensor Parallel Calculations（4 分）

讲义第 44–45 页。使用式 (47)–(51) 的 TP 策略：W₁/W₂ 按输出维分片，W₃ 按输入维分片。

#### (a) 反向公式

**题目**：给定形状为 (B, D) 的 dy，写出此 TP 策略的反向过程。W₁⁽ⁱ⁾、W₂⁽ⁱ⁾ 的形状为 (D, D_FF/N_TP)，W₃⁽ⁱ⁾ 的形状为 (D_FF/N_TP, D)。使用 dy、分片权重、前向保存的 x、x₁⁽ⁱ⁾、x₂⁽ⁱ⁾、z⁽ⁱ⁾、y⁽ⁱ⁾、通信原语及必要中间变量，最终产生各设备的 dW₁⁽ⁱ⁾、dW₂⁽ⁱ⁾、dW₃⁽ⁱ⁾ 与 dx。可参考第 8.2 节的非分片反向过程。

**交付要求**：描述反向传播的一组公式。

**答案**

【待填写公式、变量定义及形状】

#### (b) 前向与反向计算量

**题目**：N_TP 路 TP 的前向和反向分别需要多少 FLOPs？

**交付要求**：两个含 B、D、D_FF、N_TP 的表达式，各附一句理由。

**答案**

- 前向表达式与理由：【待填写】
- 反向表达式与理由：【待填写】

#### (c) 前向与反向通信时间

**题目**：N_TP 路 TP 的前向与反向通信分别耗时多久？

**交付要求**：两个表达式，使用 B、D、D_FF、N_TP、W 中所需变量，各附一句理由。

**答案**

- 前向表达式与理由：【待填写】
- 反向表达式与理由：【待填写】

#### (d) 可扩展设备数

**题目**：固定其他参数，反向与前向分别允许 N_TP 增加到什么程度而不成为通信瓶颈？

**交付要求**：两个一侧为 N_TP 的不等式，使用 B、D、D_FF、C、W 中所需变量，各附一句理由。

**答案**

- 反向约束与理由：【待填写】
- 前向约束与理由：【待填写】

### fsdp_tp_calcs — 2D Parallelism Calculations（6 分）

讲义第 45–46 页。采用第 8.5 节和式 (52)–(59) 的二维配置，总设备数 N = N_TP × N_FSDP。

#### (a) 前向计算量

**题目**：N_FSDP 路 FSDP 与 N_TP 路 TP 组合时，前向需要多少 FLOPs？

**交付要求**：含 B、D、D_FF、N_FSDP、N_TP 的表达式及一句理由。

**答案**

- 表达式：【待填写】
- 理由：【待填写】

#### (b) 两轴通信可重叠

**题目**：假设 FSDP 轴与 TP 轴通信可相互重叠，前向通信耗时是多少？

**交付要求**：使用 B、D、D_FF、N_FSDP、N_TP、W 中所需变量的表达式及一句理由。讲义提示以两轴通信代价的 max 表达。

**答案**

- 表达式：【待填写】
- 理由：【待填写】

#### (c) 可重叠时的最优扩展规模

**题目**：最优选择 N_TP 和 N_FSDP 时，总设备数 N 最大能增加到什么程度，而前向仍未成为通信瓶颈？

**交付要求**：一侧为 N 的不等式，使用 B、D、D_FF、C、W 中所需变量，附若干句解释与公式。

**答案**

- 最优配置推导：【待填写】
- N 的约束：【待填写】
- 解释：【待填写】

#### (d) 两轴通信不可重叠

**题目**：假设两轴共享网络资源而不能相互重叠。在最优 N_TP、N_FSDP 下，N 最大能增加到什么程度而前向未成为通信瓶颈？无需考虑把 N_TP 和 N_FSDP 截断为整数。

**交付要求**：一侧为 N 的不等式，使用 B、D、D_FF、C、W 中所需变量，附若干句解释与公式。

**答案**

- 推导：【待填写】
- N 的约束：【待填写】
- 解释：【待填写】

## 9. Leaderboard

### leaderboard — Leaderboard: Fastest Training Step（10 分）

讲义第 46–48 页。

**题目**

优化约 8B 模型的完整训练步骤，计时覆盖前向、loss、反向和 AdamW 更新。不得改变模型输入 / 输出行为，需通过正确性测试，且实现必须为自己编写，不得使用或复制已有实现。

计时边界还需对照第 47 页的基准函数：其训练步骤包含梯度清理，并按给定方式计算和归约 cross-entropy。调试时允许缩短测量与预热时间；正式记录须注明采用的设置。前 5–10 名提交会被复核正确性与性能。

指定条件：

| 参数 | 要求 |
| --- | --- |
| GPU | 2 × B200 |
| Batch size | 2 |
| Context length | 32768 |
| Vocab size | 151936 |
| d_model / d_ff | 4096 / 11008 |
| 层数 / 头数 | 34 / 32 |
| dtype / masking | BF16 / causal |
| 正式计时配置 | 讲义 do_bench：rep 30000 ms，warmup 10000 ms |
| 冷缓存运行限制 | 从空 PyTorch / Triton 缓存开始，benchmark 必须在 10 分钟内完成 |
| 讲义基线期望 | 超过朴素基线的性能（完整步骤少于 10 秒） |

**交付要求**：最佳完整训练步骤墙钟时间。讲义给出的[排行榜仓库](https://github.com/stanford-cs336/assignment2-systems-leaderboard)；提交结果与正式硬件设置对应。

**答案**

- 是否完成指定硬件实验：【待填写】
- 实际设备与正式配置的差异：【待填写】
- 最佳完整训练步骤耗时：【待填写，注明 ms 或 s】
- 从空缓存开始的总运行时间：【待填写】
- 正确性测试结果：【待填写】
- 对应代码版本与原始日志：【待填写】
- 排行榜提交记录（如有）：【待填写】

## 附录：原模板的 PDF 与模板逐题核对记录（2026-09-16）

> 以下为原模板已有记录，本次仅核对所附 Part 2 的原页码 2–10，未重新核验其余章节。

核对依据：本地 2026 版英文讲义 Version 26.1.3。共 27 道计分题，题框标示分值合计 137 分（仅为本文件题框分值之和，不推断课程最终计分规则）；58 个必做题作答单元，加 1 个可选记录区。下表的“对应”表示题号、分值、字母子题及答案区匹配，且已人工对照题意、实验设置和交付要求；模板仍为中文摘要，不替代讲义的公式、示例和算法正文。

| PDF Problem | PDF 页码 | 分值 | 子题 | 模板作答单元数 | 核对结果 |
| --- | --- | --- | --- | --- | --- |
| `benchmarking_script` | 3–4 | 4 | (a)、(b)、(c) | 3 | 对应 |
| `nsys_profile` | 6 | 5 | (a)、(b)、(c)、(d)、(e) | 5 | 对应 |
| `mixed_precision_accumulation` | 7–8 | 1 | 无字母子题 | 1 | 对应 |
| `benchmarking_mixed_precision` | 8–9 | 2 | (a)、(b)、(c) | 3 | 对应 |
| `memory_profiling` | 9–10 | 4 | (a)、(b)、(c)、(d)、(e)、(f) | 6 | 对应 |
| `gradient_checkpointing` | 15 | 4 | (a)、(b) | 2 | 对应 |
| `pytorch_attention` | 16 | 2 | (a) | 1 | 对应 |
| `torch_compile` | 16–17 | 2 | (a)、(b) | 2 | 对应 |
| `flash_forward` | 26–27 | 15 | (a)、(b)、(c) | 3 | 对应 |
| `flash_backward` | 28 | 5 | 无字母子题 | 1 | 对应 |
| `flash_benchmarking` | 28 | 5 | (a) | 1 | 对应 |
| `distributed_communication_single_node` | 32 | 5 | 无字母子题 | 1 | 对应 |
| `naive_ddp` | 33 | 5 | 无字母子题 | 1 | 对应 |
| `naive_ddp_benchmarking` | 33 | 3 | 无字母子题 | 1 | 对应 |
| `minimal_ddp_flat_benchmarking` | 34 | 2 | 无字母子题 | 1 | 对应 |
| `ddp_overlap_individual_parameters` | 35–36 | 5 | 无字母子题 | 1 | 对应 |
| `ddp_overlap_individual_parameters_benchmarking` | 36 | 1 | (a)、(b) | 2 | 对应 |
| `optimizer_state_sharding` | 37 | 15 | 无字母子题 | 1 | 对应 |
| `optimizer_state_sharding_accounting` | 38 | 5 | (a)、(b)、(c) | 3 | 对应 |
| `fsdp` | 38–39 | 15 | 无字母子题 | 1 | 对应 |
| `fsdp_accounting` | 39 | 5 | (a)、(b) | 2 | 对应 |
| `alternate_ring_all_reduce` | 41 | 1 | 无字母子题 | 1 | 对应 |
| `data_parallel_calcs` | 42 | 3 | (a)、(b)、(c) | 3 | 对应 |
| `fsdp_calcs` | 43–44 | 3 | (a)、(b)、(c) | 3 | 对应 |
| `tp_calcs` | 44–45 | 4 | (a)、(b)、(c)、(d) | 4 | 对应 |
| `fsdp_tp_calcs` | 46 | 6 | (a)、(b)、(c)、(d) | 4 | 对应 |
| `leaderboard` | 46–48 | 10 | 无字母子题 | 1 | 对应 |

### 原模板核对修订

- 未发现遗漏的计分题或字母子题；补充默认 FP32 基线、各 profile 分别作答的说明，以及前向加反向合计时间的记录位置。
- 修正 checkpointing (a) 的草图交付：需在报告中直接展示自己编写的草图，不能仅列文件位置。
- 明列 FlashAttention 前向的接口、执行约束和精度指导，并标注 PDF 与仓库 Triton adapter 名称不一致。
- 补充 Optimizer 父类构造、动态 parameter group、FSDP 小层不分片和公开接口要求。
- 补充 TP 反向题给定的分片形状、Leaderboard 的计时范围与提交入口。
- 保留可选 Triton 反向区；Weighted Sum 教学示例没有独立 writeup 题。
- 环境、日志与辅助表格属于记录工具；他种硬件实测不自动满足指定硬件条件。

## 附录：材料索引（可选）

| 题号 | 实现文件 | 原始结果 / 日志 | 图片 / Profile | 状态 |
| --- | --- | --- | --- | --- |
| benchmarking_script | `cs336_systems/benchmark.py`（未附） | 原模板已有计时 | 未附原始日志 | 分析已补齐；预热对照仅 small |
| nsys_profile | 未附 | `nsys_profile_tables.md` | 六份 medium/large `.nsys-rep` | 数字已整理；需修正统计口径 |
| mixed_precision_accumulation | 讲义四组代码 | NumPy 同精度复核 | 不适用 | 待原 PyTorch 代码确认 |
| memory_profiling | 未附 | 未附 | 缺 xl snapshot / 截图 | 理论推导已写；实验待补 |

## 参考文献

1. CS336 Assignment 2，所附 `cs336_assignment2_systems_part2(1).pdf`，原页码 2–10：本次题意、配置及公式依据。
2. 所附 `nsys_profile_tables.md`，2026-09-16：六组 profile 的数字汇总及过滤限制；本稿未重新导出原始报告。
3. 原 `writeup_template.md` 已填实验记录：FP32 benchmarking 与 warmup 数据来源。
4. [PyTorch 2.11 Automatic Mixed Precision](https://docs.pytorch.org/docs/2.11/amp.html#cuda-op-specific-behavior)：ToyModel 的 CUDA autocast dtype 推导。


## 本次补测清单（第 2 章）

1. **Nsight 统计修正**：六组报告分别按 forward 内 CUDA launch 归属统计完整 GPU 工作；补 forward+backward（不含 optimizer）的 kernel 排名与次数、其余五组非 GEMM 明细、attention 子操作 GPU 时间。先在已有 `.nsys-rep` 中重新导出，无需为了改统计口径立即重跑。
2. **前向对照**：补 medium@1024、medium@2048、large@256、large@1024 无 profiler 同步 forward 计时；inference 与训练图 forward 分开。提供 medium@4096 / large@2048 的容量探测日志以确认最大长度。
3. **预热及混合精度**：需要完整覆盖时补其余模型 warmup 0/1/2；运行讲义四组 PyTorch 累加代码；补各模型 BF16 forward/backward 与 OOM 状态。
4. **显存实验**：补 xl@128/2048、FP32/BF16、inference/train 的峰值和 snapshot；提供两张要求的时间线，以及最大分配栈和单 block 保存张量/梯度分析截图。
5. **实现与身份记录**：附脚本/模型源码、commit、姓名学号；核对实际 inference 开关、计时边界、TF32 设置及 memory profiling 实现。第 3–9 章原模板尚未完成，不在本次 Part 2 附件覆盖范围内。
