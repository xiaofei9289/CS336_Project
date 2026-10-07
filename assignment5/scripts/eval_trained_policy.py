"""GSM8K eval for train_grpo checkpoints (single-GPU vLLM).

Example:
  .venv/bin/python scripts/eval_trained_policy.py \\
    --model results/standard_seed_0/checkpoint_final \\
    --prompt r1_zero --gpu 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="checkpoint dir (save_pretrained)")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument("--prompt", choices=("question_only", "r1_zero", "r1_zero_three_shot"), default="r1_zero")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "cs336_alignment/prompting_baselines.py"),
        "--model",
        str(args.model),
        "--dataset",
        str(args.dataset),
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
        "--gpu",
        str(args.gpu),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
