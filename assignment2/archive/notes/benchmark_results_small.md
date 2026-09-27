# benchmarking_script small（b）（c）

- 机器：台式机 NVIDIA GeForce RTX 3060 Ti
- 环境：`C:\Users\Administrator\cs336\Assignment02\.venv\Scripts\python.exe`（`torch 2.11.0+cu128`）
- **不要**在该目录用 `uv run`（会按仓库 `uv.lock` 把 CUDA 版卸掉、装回 CPU 版）
- 脚本：`cs336_systems/benchmark_lm.py`（与 Mac 哈希一致；交作业时文件名为 `benchmark.py`）
- 日期：2026-09-16

## 共同超参（表 1 small）

batch 4，context 512，vocab 10000，d_model 768，num_layers 12，num_heads 12，d_ff 3072，FP32，measurement_steps 10。参数量 128,625,408。

`zero_grad` 未计入计时。forward 仍走 autograd（非 `no_grad` 推理）。optimizer 为 `torch.optim.AdamW`。

## (b) warmup=5

| mode | mean (ms) | std (population, ms) | samples (ms) |
|---|---|---|---|
| forward | 82.148 | 0.455 | 81.839, 81.794, 81.840, 81.791, 81.924, 82.727, 81.909, 82.018, 82.455, 83.184 |
| forward_backward | 253.909 | 0.620 | 253.287, 253.445, 253.556, 254.474, 254.763, 252.934, 254.456, 254.337, 254.486, 253.358 |
| train | 272.734 | 0.690 | 272.406, 273.178, 271.306, 272.494, 273.571, 272.914, 271.917, 272.884, 273.002, 273.663 |

由两次独立运行相减的近似（非同一步分段）：backward ≈ 172 ms，optimizer ≈ 19 ms。

## (c) 同一配置，只改 warmup-steps

| warmup | forward mean / std | fwd+bwd mean / std | train mean / std |
|---|---|---|---|
| 0 | 98.241 / 48.148 | 273.997 / 67.332 | 294.990 / 73.271 |
| 1 | 81.916 / 0.840 | 251.980 / 0.551 | 271.240 / 1.044 |
| 2 | 82.342 / 1.335 | 253.229 / 0.679 | 272.009 / 0.654 |
| 5 | 82.148 / 0.455 | 253.909 / 0.620 | 272.734 / 0.690 |

warmup=0 时第一步明显偏慢，例如 forward 首步 242.651 ms，随后回到 ~81–82 ms。
