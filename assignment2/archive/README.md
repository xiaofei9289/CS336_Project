# 交作业后的本地留档

提交包仍按当时路径。这里只整理**本机仓库**，方便以后复跑。

## `cs336_systems/` 约定

**实现（测试 import，未改名）**

- `ddp.py` / `fsdp.py` / `flash_attention.py` / `flash_attention_triton_backward.py` / `optimizer_state_sharding.py`

**测量：一律 `benchmark_<topic>.py`**

| 现文件 | 交作业时的名字 | 用途 |
| --- | --- | --- |
| `benchmark_lm.py` | `benchmark.py` | 第 2 章 LM 计时 / checkpoint / compile / BF16 |
| `benchmark_pytorch_attention.py` | `pytorch_attention.py` | 孤立 attention 扫表 |
| `benchmark_flash_attention.py` | 同左 | Flash vs PyTorch |
| `benchmark_distributed_communication.py` | `distributed_all_reduce.py` | 单机 all-reduce |
| `benchmark_ddp.py` | `ddp_benchmark.py` | naive / flat / overlap |
| `benchmark_optimizer_sharding.py` | 同左 | ZeRO-1 显存 |
| `benchmark_fsdp.py` | 同左 | FSDP 显存 + nsys NVTX |

**检查：** `check_flash_backward.py`

**讲义原题脚本：** `mixed_precision_accumulation.py`（`python -m cs336_systems.mixed_precision_accumulation`）

## 本目录

- `new_benchmark.py`：学习重写，不是第 2 章数字来源
- `writeup_template*.md`：写报告用的草稿
- `notes/`：过程笔记（解读、学习日志、AutoDL 配置、small 结果草稿）

## 仓库其它目录

- `notes/`：只留 `a800_4gpu.md`、`nsys_profile_tables.md`
- `results/`：写入 writeup 的表；试跑与重复 csv 在 `results/smoke/`
- `mem_snapshots/`：给助教看的图；原始 pickle 在 `mem_snapshots/raw/`
- `nsys_reports/`、`nsys_stats/`：Nsight 原件与导出 csv
