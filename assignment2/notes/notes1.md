作业 2 的目标是：先量清单卡瓶颈，再自己写 kernel 和多卡并行，最后把结果写进 writeup。提交物是 `writeup.pdf` + `code.zip`（`./test_and_make_submission.sh`）。官方说明见 [assignment2-systems](https://github.com/stanford-cs336/assignment2-systems/tree/main)，仓库里已有 `CS336_Assignment2_Systems_中文翻译.pdf`。

总分大约 137 分（含 10 分 leaderboard）。建议按「可复用脚本 → 有自动测试的实现 → 只能 GPU 上测的实验 → 纸面计算 → 冲榜」这条依赖链走。

------

## 总路线（按依赖）

环境 + 基准脚本

​    → 单卡 profiling / mixed precision / memory / checkpoint

​    → Attention 基线 + torch.compile

​    → FlashAttention（PyTorch → Triton 前向 → backward → 对比）

​    → 通信测量 → naïve DDP → flatten → overlap DDP

​    → Optimizer state sharding → FSDP

​    → 第 8 节纸面计算（可穿插）

​    → Leaderboard（最后冲）

两条可以并行的线：

| 线                    | 什么时候做                | 需要什么                              |
| :-------------------- | :------------------------ | :------------------------------------ |
| A. 自动测试代码       | 随时（很多能在 CPU 上过） | `tests/adapters.py` 接到你的实现      |
| B. GPU 实验 + writeup | 有卡再集中跑              | 单卡、2 卡、最多 6 卡；Nsight Systems |

不要等所有代码写完才写 writeup。每做完一块就把表、截图、2–3 句话结论立刻记下来。

------

## 阶段 0：环境（半天）

1. `uv run python`，确认能 `import cs336_basics`。
2. 熟悉入口：你的代码写在 `cs336_systems/`，测试只认 `tests/adapters.py`。
3. 表 1 的模型规格先记下来（默认 vocab 10000、batch 4、context 512）：

| Size                              | d_model                         | d_ff                               | layers                 | heads                  |
| :-------------------------------- | :------------------------------ | :--------------------------------- | :--------------------- | :--------------------- |
| small / medium / large / xl / 10B | 768 / 1024 / 1280 / 2560 / 4608 | 3072 / 4096 / 5120 / 10240 / 12288 | 12 / 24 / 36 / 32 / 50 | 12 / 16 / 20 / 32 / 36 |

1. 装好 Nsight Systems（`nsys`）。后面几乎所有时间/通信分析都靠它。
2. 确认 GPU：Triton、混合精度、memory snapshot、DDP/FSDP 计时、leaderboard 都要 CUDA；DDP/FSDP/sharded optimizer 的 正确性测试 用 `gloo`，CPU 也能跑。

------

## 阶段 1：基准脚本（约 4 分，但后面全靠它）

题：`benchmarking_script`

先写一个命令行脚本，能切模型大小、是否 warmup、只 forward / forward+backward / 完整 step（含 AdamW）。计时必须 `torch.cuda.synchronize()`。

立刻用它交：

- warmup 5 + 测 10 步的均值/标准差
- 关掉 warmup、以及 warmup=1/2 的对比

这套脚本后面会反复加开关：BF16、memory snapshot、NVTX、checkpoint、DDP、FSDP。先把 CLI 做稳，比先写 FlashAttention 更省时间。

------

## 阶段 2：单卡系统分析（约 16 分）

这是 writeup 大头，建议 GPU 一到手就做，别拖到期末。

| 题                             | 分   | 你要交什么                                                   |
| :----------------------------- | :--- | :----------------------------------------------------------- |
| `nsys_profile`                 | 5    | 两种模型 × 三种 context；forward 总时间、最重 kernel、非 matmul kernel、完整 step 里 matmul 占比、attention 里 softmax vs matmul |
| `mixed_precision_accumulation` | 1    | 四段累加代码的数值差异（CPU 就能做）                         |
| `benchmarking_mixed_precision` | 2    | ToyModel 各张量 dtype；BF16 autocast 扫表 1                  |
| `memory_profiling`             | 4    | xl @ 128/2048 的 memory_viz 时间线、峰值、混合精度是否省显存、residual 张量大小推导 |

关键习惯：用 NVTX 把 warmup / forward / backward / optimizer 标出来，`nsys` 才能按区间过滤。

------

## 阶段 3：Activation checkpointing（4 分）

题：`gradient_checkpointing`

- (a) 纸面：N*N* 层怎么 nested checkpoint 才能把激活峰值压到最低（渐近 memory / compute）。
- (b) 实验：xl、batch 4、seq 2048，只允许一层 recomputation（不能嵌套），选最优 block 粒度，并和相邻粒度比峰值。

这题和后面 leaderboard 的「显存装不下」直接相关，但实现本身不是 pytest 必过项。

------

## 阶段 4：Attention 优化（约 29 分，核心实现）

建议顺序，不要一上来写 Triton：

1. `pytorch_attention`（2）
   batch=8、无 head 维，扫 d*d* 和 seq。很多配置会 OOM——OOM 本身就是答案。算清 N2*N*2 注意力矩阵为什么吃显存。
2. `torch_compile`（2）
   先 compile 单独 attention，再 compile 整个 Transformer，和 naive 比。
3. `flash_forward`（15）——作业里最重的一块
   - 先写 纯 PyTorch 的 `autograd.Function`（慢没关系，用来对照）
     测试：`uv run pytest -k test_flash_forward_pass_pytorch`
   - 再写 Triton 前向（需要 GPU）
     测试：`uv run pytest -k test_flash_forward_pass_triton`
   - 最后加 `is_causal`
4. `flash_backward`（5）
   作业要求 backward 用 PyTorch + `torch.compile`，不是 Triton。
   测试：`uv run pytest -k test_flash_backward`
5. `flash_benchmarking`（5）
   `triton.testing.do_bench`，seq 128–65536、d 16–128、bf16/fp32，报 forward / backward / e2e。

Triton backward（§4.2.3）是 可选，只为 leaderboard 加速，不影响必交分。

Adapters 对应：

- `get_flashattention_autograd_function_pytorch`
- `get_flashattention_autograd_function_triton`

------

## 阶段 5：数据并行（约 21 分）

先通信、再正确性、再计时。

1. `distributed_communication_single_node`（5）
   单机 all-reduce：1MB–1GB × 2/4/6 GPU。先 Gloo 调试，再 NCCL 计时。
2. `naive_ddp`（5）
   broadcast 参数 → 各卡算自己的梯度 → 再 all-reduce。
   测试：`uv run pytest tests/test_ddp.py`（建议连跑 5 次，抓竞态）。
3. `naive_ddp_benchmarking`（3）
   1 节点 × 2 GPU，xl：每步总时间 + 通信占比。
4. `minimal_ddp_flat_benchmarking`（2）
   把梯度拼成一块再一次 all-reduce，和「每个参数一次」对比。
5. `ddp_overlap_individual_parameters`（5）+ 计时（1）
   这才是最终要接到 adapter 的 DDP：反向过程中异步通信，`finish_gradient_synchronization()` 再 `optimizer.step()`。Nsight 截图要能看出 overlap vs 不 overlap。

Adapters：`get_ddp`、`ddp_on_after_backward`。
测试会检查 rank0 广播、和全量数据单卡训练权重一致、以及 tied weights。

------

## 阶段 6：Optimizer sharding（20 分）

题：`optimizer_state_sharding`（15）+ accounting（5）

每张卡只持有约 1/world_size1/world_size 的 optimizer state，step 后再把更新后的参数同步回去。

测试：`uv run pytest tests/test_sharded_optimizer.py`（同样建议连跑多次）。

Accounting 三问（1 节点 × 2 GPU、xl）：

- 初始化后 / step 前 / step 后的峰值显存拆解
- 每步耗时有没有变慢
- 和 ZeRO-1 差在哪（通信量和显存）

Adapter：`get_sharded_optimizer`。

------

## 阶段 7：FSDP（20 分，后半核心）

题：`fsdp`（15）+ `fsdp_accounting`（5）

作业要求：

- 切 Linear / Embedding，Norm 不要切
- forward/backward 前 all-gather 权重，backward 后 reduce-scatter 梯度
- 用完立刻释放 gather 出来的完整权重
- 提前两层 prefetch，避免计算空等
- `compute_dtype` 非空时：通信和计算用低精度，master weight 和 optimizer 仍 FP32

测试：`uv run pytest tests/test_fsdp.py`（fp32 和 fp16 各跑；建议连跑多次）。

Adapters：`get_fsdp`、`fsdp_on_after_backward`、`fsdp_gather_full_params`。

Accounting：对照第 6 节估算省了多少显存；xl 双卡 Nsight 看 all-gather 是否赶在 forward 前完成。

------

## 阶段 8：纸面并行分析（17 分，可穿插）

不依赖 GPU，FlashAttention 卡住时就做这节。

| 题                          | 分   |
| :-------------------------- | :--- |
| `alternate_ring_all_reduce` | 1    |
| `data_parallel_calcs`       | 3    |
| `fsdp_calcs`                | 3    |
| `tp_calcs`                  | 4    |
| `fsdp_tp_calcs`             | 6    |

统一用作业给的符号：B,D,DFF,NDP/NFSDP/NTP,W*B*,*D*,*D**F**F*,*N**D**P*/*N**F**S**D**P*/*N**T**P*,*W*。先画一张「每层通信什么、算什么」的示意图，再写公式，不容易漏因子。

------

## 阶段 9：Leaderboard（10 分，最后冲）

配置故意很难装进显存：batch 2、seq 32768、vocab 151936、d_model 4096、34 层，两张 B200 上完整 train step（forward + loss + backward + AdamW）。基线要优于 10 秒，空 cache 下 10 分钟内跑完。

提交：[assignment2-systems-leaderboard](https://github.com/stanford-cs336/assignment2-systems-leaderboard)

作业给的方向（自己实现，不能抄现成 FA）：调 tile / compile、fused AdamW、fused LM-head+CE、Triton backward、causal 早停、TMA、必要时才 checkpoint。先保证能跑完一步，再抠时间。

------

## 建议日历（按 3–4 周估）

| 周        | 做什么                                                      | 累计可交分（不含榜） |
| :-------- | :---------------------------------------------------------- | :------------------- |
| 第 1 周   | 环境 + 基准脚本 + 全部 profiling / mixed precision / memory | ~16                  |
| 第 1–2 周 | checkpoint + attention 基线 + compile；同时做第 8 节计算    | ~29                  |
| 第 2 周   | FlashAttention PyTorch → Triton 前向 → backward → 对比表    | ~58                  |
| 第 3 周   | 通信 + naïve/flat/overlap DDP                               | ~79                  |
| 第 3–4 周 | sharded optimizer + FSDP + 显存/Nsight accounting           | ~119                 |
| 机动      | writeup 排版、leaderboard                                   | +10                  |

CPU 笔记本上也能推进：Flash 的 PyTorch 版、DDP/FSDP/optimizer 正确性测试、第 8 节计算、writeup 草稿。GPU 时间留给 nsys、Triton、多卡计时、冲榜。

------