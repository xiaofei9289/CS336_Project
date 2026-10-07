# Reproduction

This document records the environment, data layout, commands, expected outputs, and the checks that have actually been run. The aggregate tables in `results/` come from a saved-artifact audit on October 7, 2026. The GPU training and kernel benchmarks were not rerun while these pages were written.

## What this checkout contains

The Assignment 5 training and evaluation code is in `assignment5/`. `tests/adapters.py` imports that code. GSM8K, MMLU, Anthropic HH, the GRPO logs, the final validation rollouts, and the SFT/DPO evaluation files are committed. OLMo-2-0425-1B weights, Llama-3.1-8B weights, the SFT and DPO checkpoints, and the UltraChat training file are not. `results/source_manifest.json` lists 179 SHA-256 hashes. On October 7, 2026, every one of those hashes matched the file at the same path on `main`.

## Environment

Assignment 5 declares Python `>=3.12,<3.13` in `assignment5/pyproject.toml`. The recorded remote training environment is separate from that file and was not frozen as a lock of the GPU machine.

| Experiment | Recorded runtime |
|---|---|
| GRPO | 2026-10-02, AutoDL, 2× RTX PRO 6000 Blackwell (96GB). Policy on physical GPU 0. vLLM on physical GPU 1. Local weights `OLMo-2-0425-1B`. |
| SFT | One RTX PRO 6000. Llama-3.1-8B, BF16, FlashAttention 2. |
| DPO | Two GPUs. Policy on `cuda:0`, frozen reference on `cuda:1`. |
| DDP and optimizer sharding | 2026-09-28 and 2026-09-29, AutoDL, 2× RTX PRO 6000 Blackwell Server Edition, PyTorch 2.11.0+cu128, NCCL, conda base. |

Install the Assignment 5 environment from `assignment5/`:

```sh
uv sync --no-install-package flash-attn
uv sync
```

`flash-attn` is required by the SFT and DPO scripts, which request `attn_implementation="flash_attention_2"`. The CPU tests below do not need it.

## Data preparation

GRPO reads JSONL objects with `question` and `answer`. The answer string keeps the GSM8K `####` final answer. Put the files at:

```text
assignment5/data/gsm8k/train.jsonl
assignment5/data/gsm8k/test.jsonl
```

Those two GSM8K files are already committed. `load_examples` keeps the first `--n-train-examples` training rows and the first `--n-val-examples` validation rows. The saved runs use 6,400 and 1,024. The 1,024 validation questions are the prefix of `test.jsonl`, and the same prefix was reused while configurations were compared.

DPO reads four gzipped Anthropic HH training files from `assignment5/data/hh/`:

```text
harmless-base.jsonl.gz
helpful-online.jsonl.gz
helpful-base.jsonl.gz
helpful-rejection-sampled.jsonl.gz
```

Those four archives are already committed. `load_hh_train` keeps single-turn pairs, shuffles them with seed 0, holds out 200 pairs, and trains on the remaining 49,188 rows.

SFT was trained on `data/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz` with validation file `test.jsonl.gz` in that directory. Those files are not in this repository. Model directories are passed on the command line. The saved runs used local copies of OLMo-2-0425-1B and Llama-3.1-8B.

## Commands

Run GRPO from `assignment5/` after `MODEL_DIR` points at OLMo-2-0425-1B. Off-policy clipped training resolves its algorithm settings from the `offpolicy_clip` recipe: train batch 8, gradient accumulation 1, 32 updates per rollout batch, clip range 0.2. The CLI default `--gradient-accumulation-steps 32` applies to on-policy `scripts/train_grpo.py`. The variant script replaces that default when the recipe sets its own accumulation.

```sh
python scripts/train_grpo_variants.py \
  --recipe offpolicy_clip \
  --model MODEL_DIR \
  --train data/gsm8k/train.jsonl \
  --val data/gsm8k/test.jsonl \
  --prompt cs336_alignment/prompts/r1_zero.prompt \
  --n-train-examples 6400 --n-val-examples 1024 \
  --num-rollout-steps 200 --rollout-batch-size 256 --group-size 8 \
  --learning-rate 1e-5 --max-tokens 512 --seed 0 \
  --policy-device cuda:0 --vllm-gpu 1 \
  --output-dir results/offpolicy_clip_seed_0
```

Repeat with `--seed 1`, `--seed 2`, and `--seed 3`. `scripts/run_all_experiments.sh offpolicy` launches the four off-policy recipes for seeds 0–3. Standard on-policy GRPO is `scripts/train_grpo.py` or `./scripts/run_all_experiments.sh standard`.

SFT, from `assignment5/`:

```sh
python scripts/train_sft.py \
  --model MODEL_DIR \
  --train data/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz \
  --val data/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz \
  --output-dir results_safety/sft_llama31_8b
```

DPO starts from that SFT checkpoint:

```sh
python scripts/train_dpo.py \
  --model results_safety/sft_llama31_8b \
  --hh-dir data/hh \
  --output-dir results_safety/dpo_llama31_8b \
  --policy-device cuda:0 \
  --ref-device cuda:1
```

DDP, from `assignment2/` on two GPUs:

```sh
python cs336_systems/benchmark_ddp.py --mode naive --world_size 2 --backend nccl
python cs336_systems/benchmark_ddp.py --mode overlap_params --world_size 2 --backend nccl
```

Optimizer sharding:

```sh
python -m cs336_systems.benchmark_optimizer_sharding
```

## Expected outputs

An off-policy clipped run writes `assignment5/results/offpolicy_clip_seed_<seed>/` with `run_config.json`, `metrics.jsonl`, and `val_rollouts_final.jsonl`. The saved final answer accuracies at step 200 on 1,024 questions are:

| Seed | Answer accuracy |
|---|---|
| 0 | 0.5380859375 |
| 1 | 0.546875 |
| 2 | 0.5458984375 |
| 3 | 0.5009765625 |
| Mean | 0.532958984375 |
| Sample SD | 0.021681373690995244 |

`run_config.json` should show `algorithm.importance_reweighting_method` = `grpo`, `algorithm.cliprange` = `0.2`, `algorithm.off_policy_updates` = `32`, and `algorithm.gradient_accumulation_steps` = `1`.

SFT and DPO benchmark summaries that match `results/sft_dpo_results.csv`:

| File | Accuracy |
|---|---|
| `mmlu_sft_alpaca.summary.json` | 8598/14042 = 0.6123059393248825 |
| `gsm8k_sft_alpaca.summary.json` | 416/1319 = 0.3153904473085671 |
| `mmlu_dpo_alpaca.summary.json` | 8470/14042 = 0.6031904287138584 |
| `gsm8k_dpo_alpaca.summary.json` | 392/1319 = 0.2971948445792267 |

The saved DDP logs report mean step times of 1225.5480 ms (`naive`) and 1022.0085 ms (`overlap_params`) over 10 measured steps. The sharding log reports rank-0 cumulative peaks of 63.756 GiB and 44.850 GiB.

The TinyStories log ends at update 40000 with validation cross-entropy 1.4019526660442352. The saved BPE run reports `elapsed_seconds=100.739`, vocabulary 10000, and 9743 merges.

## Checks completed for this repository update

| Check | Result |
|---|---|
| Headline code paths exist after copying the local Assignment 5 implementation | Confirmed for `dpo.py`, `grpo.py`, `grpo_offpolicy.py`, `training_runtime.py`, and `scripts/train_{sft,dpo,grpo,grpo_variants}.py` |
| `offpolicy_clip` mean recomputed from `results/grpo_seed_results.csv` | 4 values, mean 0.532958984375, sample SD 0.021681373690995244 |
| DDP and sharding logs in `assignment2/results/` match `results/systems_results.csv` | Confirmed against the three `rtxpro6000_*.txt` files |
| TinyStories final validation cell | `assignment1/runs/tinystories/training_log.csv` row 40000, third numeric field 1.4019526660442352 |
| BPE elapsed time | `assignment1/runs/tinystories_bpe/optimize_console.txt` reports 100.739 seconds and 9743 merges |
| Artifact hashes | All 179 paths in `results/source_manifest.json` match the blobs on `main` |
| CPU tests on the Assignment 5 tree | October 7, 2026, local Assignment 5 virtualenv, `PYTHONPATH` pointed at this checkout, no GPU. `tests/test_grpo.py`, `tests/test_dpo.py`, and `tests/test_metrics.py`: 24 passed. `tests/test_rollout_batching.py` and `tests/test_train_integration.py`: 4 passed. |

## Checks not rerun

- The 56 GRPO training runs, the SFT run, and the DPO run.
- vLLM generation for MMLU, GSM8K, AlpacaEval, and SimpleSafetyTests.
- The two-GPU DDP and optimizer-sharding benchmarks.
- A controlled BPE speed comparison. The profiled baseline and the optimized timing run did not use the same instrumentation.
- A fresh download of model weights. The committed GSM8K, HH, and evaluation files are already in the tree.

## Limits that stay attached to the numbers

The GRPO validation split was reused during configuration selection. Off-policy runs take 32 optimizer updates per rollout batch, and standard GRPO takes one. Clipped and unclipped off-policy means are close. DPO is below the saved SFT benchmark scores under this recipe. The sharding measurement omits gradient all-reduce, so it does not by itself show the same convergence in synchronized training.
