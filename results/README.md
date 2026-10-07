# Result tables

Numbers in this directory were checked on October 7, 2026. This pass did not rerun the GPU experiments. The 179 hashed source files are committed under `assignment5/`, and every hash in `source_manifest.json` matches `main`. Model weights and the SFT/DPO checkpoints are not committed.

## Files

| File | Contents |
|---|---|
| `key_results.csv` | Headline rows with model, baseline, dataset, seeds, GPU, and evaluation conditions |
| `grpo_config_summary.csv` | Mean and sample SD (`ddof=1`) for all 14 GRPO configurations |
| `grpo_seed_results.csv` | One final accuracy per seed |
| `sft_dpo_results.csv` | MMLU and GSM8K counts for base, SFT, and DPO |
| `systems_results.csv` | DDP step time and optimizer-sharding memory and time |
| `source_manifest.json` | Hashes of the local files used in the audit |

## Shared GRPO conditions

Every row in `grpo_config_summary.csv` uses the same evaluation contract:

| Field | Value |
|---|---|
| Model | OLMo-2-0425-1B |
| Dataset | GSM8K. Training uses the first 6,400 rows of `train.jsonl`. Validation uses the first 1,024 rows of `test.jsonl`. |
| Seeds | 0, 1, 2, and 3 |
| Final step | 200 |
| GPU | Two RTX PRO 6000 Blackwell GPUs. The policy is on `cuda:0`. vLLM is on GPU 1. |
| Sampling | Rollout batch 256, group size 8, temperature 1, top-p 1, max tokens 512 |
| Learning rate | `1e-5`, except `lr_5e-6` and `lr_3e-5` |
| Prompt | `r1_zero`, except `prompt_question_only` and `prompt_r1_zero_three_shot` |

`offpolicy_clip` is the 53.3% row. Its four final accuracies are 0.5380859375, 0.546875, 0.5458984375, and 0.5009765625. The mean is 0.532958984375. The sample standard deviation is 0.021681373690995244. The recipe uses importance reweighting `grpo`, clip range 0.2, train batch 8, gradient accumulation 1, and 32 off-policy updates per rollout batch. On-policy `standard` uses one optimizer update per rollout batch, so the comparison is not compute-matched.

The 1,024 validation questions were scored repeatedly while configurations were selected. The mean is a validation score, not an untouched final test.

## SFT and DPO

`sft_dpo_results.csv` stores `correct_answers / scored_questions`.

| Evaluation | Model and prompt | GPU and decoding |
|---|---|---|
| `mmlu_zero_shot`, `gsm8k_zero_shot` | Llama-3.1-8B base, zero-shot prompt | Greedy, temperature 0, top-p 1, max tokens 512 |
| `*_sft_alpaca` | Full-parameter SFT, Alpaca prompt | One RTX PRO 6000. One epoch, sequence length 512, microbatch 2, accumulation 16, lr `2e-5`, AdamW, BF16 |
| `*_dpo_alpaca` | DPO from the SFT checkpoint, Alpaca prompt | Policy on `cuda:0`, frozen reference on `cuda:1`. HH single-turn data, 200 validation pairs, 49,188 training rows, beta 0.1, lr `1e-6`, RMSprop, effective batch 64 |

Base and SFT use different prompts, so the base-to-SFT gap mixes the prompt change with training.

## Systems

`systems_results.csv` matches the logs in `assignment2/results/rtxpro6000_naive_ddp.txt`, `rtxpro6000_overlap_ddp.txt`, and `rtxpro6000_optimizer_sharding_accounting.txt`. Both machines are two RTX PRO 6000 Blackwell Server Edition GPUs with PyTorch 2.11.0+cu128 and NCCL. The sharding microbenchmark does not all-reduce gradients.

## Check a mean locally

From the repository root, with Python 3:

```sh
python -c '
import csv, statistics
rows = list(csv.DictReader(open("results/grpo_seed_results.csv")))
vals = [float(r["answer_accuracy"]) for r in rows if r["configuration"]=="offpolicy_clip"]
print(len(vals), statistics.mean(vals), statistics.stdev(vals))
'
```

Expected output:

```text
4 0.532958984375 0.021681373690995244
```
