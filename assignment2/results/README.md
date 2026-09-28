# results

写入 writeup 的正式表放在本目录。试跑、重复文件在 `smoke/`。

- `bench213_*.json`：2026-09-26 重测 2.1.3（`benchmarking_script`）的新结果，未覆盖旧 json
- `bench_213_logs/`：同一次运行的终端日志（含 10B OOM）

- `a800_all_reduce_results.csv`：单机 all-reduce 正式表（A800，2/4 进程）
- `rtxpro6000_all_reduce_results.csv`：2026-09-28 6×RTX PRO 6000 的 NCCL all-reduce（2/4/6 进程，8 跳过）
- `rtxpro6000_naive_ddp.txt`：2026-09-28 2×RTX PRO 6000 的朴素 DDP，xl，每步与梯度通信时间
- `rtxpro6000_batch_ddp.txt`：同机同配置的展平梯度 DDP
- `rtxpro6000_overlap_ddp.txt`：同机同配置的重叠通信 DDP，只有每步时间
- `smoke/all_reduce_results.csv`：与上一份字节相同，是脚本默认文件名留下的副本
- `smoke/*_smoke.csv`：attention / Flash 试跑，正文不用
