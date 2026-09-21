# nsys_profile 数字整理（AutoDL RTX PRO 6000）

整理日期：2026-09-16；2026-09-17 增补 `nvtx_gpu_proj_sum`。(a)–(e) 的分析句写在 `writeup.md`，不要写进模板。

- 硬件：NVIDIA RTX PRO 6000 Blackwell，PyTorch 2.11.0+cu128，nsys 2026.5.1
- 配置：batch 4，FP32，`--mode train --timing total --nvtx`，warmup 1，measurement 1
- 优化器：`cs336_basics.optimizer:AdamW`
- 捕获：`--capture-range nvtx --nvtx-capture measurement -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0`
- 模型：medium（512/1024/2048），large（256/512/1024）
- 原始 `.nsys-rep`：AutoDL `/root/autodl-tmp/nsys/`（本机可放 `nsys_reports/`）
- csv：AutoDL `/root/autodl-tmp/nsys/stats/`（本机 `nsys_stats/`）

`gemm-like` 只按 kernel 名字是否含 `gemm` / `cutlass` / `sgemm` 加总，写 (d) 前请再核对。

## 读表限制

1. `*_attn_*` 的 `--filter-nvtx` 默认只取 **第一次** attention（`inst=1`）。(e) **不要**用 attn csv 当 24/36 层总和；用 `*_nvtxname_cuda_gpu_kern_sum_nvtx-name.csv` 按最内层 NVTX 前缀加总 GPU kernel。CPU 的 `*_all_nvtx_sum.csv` 只作错误口径对照。
2. `--timing total` 时 `forward` NVTX 内部没有 CUDA sync，是 CPU 推出 kernel 的墙钟。`--filter-nvtx forward` 的 kernel 表只覆盖这段 CPU 窗口，长序列上可能偏短。**(a) 不要用下面「NVTX 阶段」的 CPU forward 列**；用 2026-09-17 导出的 `*_nvtx_gpu_proj_sum.csv` 的 `:forward` Total Proj Time，并对照无 nsys、带 sync 的 Python 计时。
3. NVTX `Time(%)` 含嵌套 range，不能当「占一步的比例」。阶段用 Total Time。
4. 作业 attention softmax 多半是 `exp` + `reduce` + 除法，不要把 `cunn_SoftMax*` 当成 attention softmax。

## 无 nsys 的 Python 对照（已有记录）

| 配置 | 来源 | 时间 (ms) |
| --- | --- | ---: |
| medium @512 stages-forward | writeup (b) | 47.538 |
| medium @512 forward-total | writeup (b) | 46.573 |
| medium @512 train-total | writeup (b) | 164.884 |
| medium @1024 forward-total | 2026-09-17 无 nsys | 136.481 ± 0.102 |
| medium @1024 stages-forward | 2026-09-17 无 nsys | 136.447 ± 0.058 |
| medium @2048 forward-total | 2026-09-17 无 nsys | 397.178 ± 0.091 |
| medium @2048 stages-forward | 2026-09-17 无 nsys | 397.072 ± 0.068 |
| large @256 forward-total | 2026-09-17 无 nsys | 57.095 ± 0.069 |
| large @256 stages-forward | 2026-09-17 无 nsys | 57.200 ± 0.024 |
| large @512 stages-forward | writeup (b) | 111.164 |
| large @512 forward-total | writeup (b) | 110.682 |
| large @512 train-total | writeup (b) | 380.215 |
| large @1024 forward-total | 2026-09-17 无 nsys | 305.633 ± 0.041 |
| large @1024 stages-forward | 2026-09-17 无 nsys | 305.701 ± 0.061 |
| medium @2048 train-total | 探测 1 step | 1260.158 |
| large @1024 train-total | 探测 1 step | 934.667 |

六组前向对照已齐。不要拿 nsys 包着跑的墙钟当基准。

## (a) GPU 投影前向（2026-09-17）

来源：`nsys_stats/{medium,large}_{len}_nvtx_gpu_proj_sum.csv`。writeup (a) 用 `:forward` 的 Total Proj Time，不用 CPU Range Time。

| 配置 | GPU 投影 forward (ms) | GPU ops | CPU Range Time forward (ms) | `:backward` 投影 (ms) / ops | measurement 投影 (ms) | optimizer 投影 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| medium 512 | 49.591 | 1424 | 48.779 | 0.0007 / 1 | 184.961 | 37.197 |
| medium 1024 | 138.267 | 1376 | 35.437 | 0.0007 / 1 | 444.068 | 26.338 |
| medium 2048 | 398.605 | 1376 | 101.614 | 0.0007 / 1 | 1264.658 | 30.510 |
| large 256 | 58.430 | 2060 | 48.367 | 0.0007 / 1 | 217.939 | 58.612 |
| large 512 | 113.023 | 2060 | 73.157 | 0.0007 / 1 | 381.974 | 52.619 |
| large 1024 | 305.226 | 1916 | 140.558 | 0.0007 / 1 | 924.768 | 42.354 |

`:backward` 六组都只有 1 个 GPU op，不能当反向或前向+反向。`measurement` 含 optimizer。

## NVTX 阶段（CPU 墙钟，ms；旧口径，勿当 (a) 结论）

来源：`*_all_nvtx_sum.csv`。这是 CPU 提交墙钟，不是 GPU 前向跨度。

| 配置 | measurement | forward | backward | optimizer |
| --- | ---: | ---: | ---: | ---: |
| medium 512 | 186.824 | 48.779 | 66.525 | 69.173 |
| medium 1024 | 445.713 | 35.437 | 283.753 | 124.318 |
| medium 2048 | 1266.188 | 101.614 | 830.212 | 332.540 |
| large 256 | 220.379 | 48.367 | 88.474 | 80.956 |
| large 512 | 384.336 | 73.157 | 206.144 | 101.956 |
| large 1024 | 927.148 | 140.558 | 605.269 | 178.245 |

## GPU kernel 合计（名字粗分类）

| 配置 | 前向 GPU 合计 | 前向 gemm-like | 前向非 gemm | 整步 GPU 合计 | 整步 gemm-like | 整步非 gemm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| medium 512 | 47.113 ms | 79.5% | 20.5% | 159.422 ms | 61.2% | 38.8% |
| medium 1024 | 34.883 ms | 54.1% | 45.9% | 431.122 ms | 45.1% | 54.9% |
| medium 2048 | 101.132 ms | 40.9% | 59.1% | 1245.825 ms | 34.4% | 65.6% |
| large 256 | 46.392 ms | 86.9% | 13.1% | 196.265 ms | 61.7% | 38.3% |
| large 512 | 71.894 ms | 79.8% | 20.2% | 365.618 ms | 60.8% | 39.2% |
| large 1024 | 139.807 ms | 59.1% | 40.9% | 918.900 ms | 50.0% | 50.0% |

占比相对该表 GPU 时间之和，不是 wall clock。左半「前向」来自 `--filter-nvtx forward`，**不要**再当 writeup (d)。(d) 用下面的 inference vs train。

## (d) inference vs train GEMM（2026-09-17）

来源：`*_inference_all_cuda_gpu_kern_sum.csv` 与 `*_all_cuda_gpu_kern_sum.csv`。gemm-like = 名字含 `gemm` / `cutlass` / `sgemm`。inference 合计按 csv Total Time 加总。

| 配置 | inference GPU 合计 | inference gemm | inference 非 gemm | train GPU 合计 | train gemm | train 非 gemm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| medium 512 | 47.582 ms | 79.8% | 20.2% | 159.422 ms | 61.2% | 38.8% |
| medium 1024 | 137.991 ms | 53.8% | 46.2% | 431.122 ms | 45.1% | 54.9% |
| medium 2048 | 402.305 ms | 40.7% | 59.3% | 1245.825 ms | 34.4% | 65.6% |
| large 256 | 56.040 ms | 87.4% | 12.6% | 196.265 ms | 61.7% | 38.3% |
| large 512 | 113.909 ms | 80.1% | 19.9% | 365.618 ms | 60.8% | 39.2% |
| large 1024 | 307.475 ms | 60.4% | 39.6% | 918.900 ms | 50.0% | 50.0% |

## (b) 累计 GPU 时间第一的 kernel

| 配置 | 前向第一 | 前向 ms / 次数 | 整步第一 | 整步 ms / 次数 |
| --- | --- | --- | --- | --- |
| medium 512 | cutlass `sgemm_256x128_…_tn` | 26.245 / 144 | 同一个 tn gemm | 26.245 / 144 |
| medium 1024 | cutlass `sgemm_128x256_…_tn` | 15.698 / 45 | 同一个 tn gemm | 60.840 / 169 |
| medium 2048 | cutlass `sgemm_128x256_…_tn` | 21.441 / 18 | 变了：elementwise Mul | 148.567 / 388 |
| large 256 | cutlass `sgemm_128x256_…_tn` | 26.918 / 89 | 变了：cutlass `sgemm_256x128_…_nn` | 36.128 / 253 |
| large 512 | cutlass `sgemm_128x256_…_tn` | 53.701 / 164 | 同一个 tn gemm | 83.545 / 253 |
| large 1024 | cutlass `sgemm_128x256_…_tn` | 72.082 / 116 | 同一个 tn gemm | 160.240 / 253 |

## (c) 前向非 gemm、Time(%) ≥ 1%（medium 512）

elementwise Mul / Add / Div、`where`、`exp_kernel`、`reduce Max`、copy。其它 5 组在对应

`*_fwd_cuda_gpu_kern_sum_nvtx=forward.csv`

里用同样规则列。长 context 时前向非 gemm 占比升高。

## (e) 全层 GPU kernel（`cuda_gpu_kern_sum:nvtx-name`，ms）

来源：`nsys_stats/*_nvtxname_cuda_gpu_kern_sum_nvtx-name.csv`。按启动 API 的最内层 NVTX 归类；medium Instances=24，large=36。matmul 只计 gemm；softmax 为 `attention_softmax` 下全部 kernel。不要用 `*_attn_*` 的 `inst=1` 时间窗。

| 配置 | QKᵀ GEMM | PV GEMM | 两个 matmul | softmax GPU | softmax/matmul | 层数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| medium 512 | 1.681 | 1.333 | 3.014 | 3.546 | 1.177 | 24 |
| medium 1024 | 6.532 | 6.000 | 12.532 | 35.766 | 2.854 | 24 |
| medium 2048 | 24.742 | 19.574 | 44.317 | 140.694 | 3.175 | 24 |
| large 256 | 0.871 | 0.681 | 1.553 | 1.851 | 1.192 | 36 |
| large 512 | 2.932 | 2.623 | 5.555 | 9.538 | 1.717 | 36 |
| large 1024 | 11.872 | 10.446 | 22.318 | 66.840 | 2.995 | 36 |

## (e) 全层 NVTX 合计（CPU，ms；错误口径，勿当结论）

来源：`*_all_nvtx_sum.csv`。

| 配置 | attention 总 | scores_matmul | attention_final_matmul | 两个 matmul 之和 | attention_softmax | 层数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| medium 512 | 14.030 | 3.220 | 2.475 | 5.695 | 4.584 | 24 |
| medium 1024 | 10.377 | 2.243 | 1.841 | 4.084 | 3.774 | 24 |
| medium 2048 | 54.246 | 1.269 | 1.756 | 3.025 | 45.263 | 24 |
| large 256 | 11.691 | 2.069 | 2.805 | 4.874 | 3.435 | 36 |
| large 512 | 29.559 | 7.372 | 3.703 | 11.075 | 10.152 | 36 |
| large 1024 | 22.751 | 4.053 | 2.980 | 7.033 | 9.259 | 36 | |

## 建议 writeup 结构

开头写清两种模型、三种长度、文件名、`measurement` 窗口；然后 6 组各写 (a)–(e)。(a) 抄 GPU 投影表，不要抄 CPU NVTX forward。
