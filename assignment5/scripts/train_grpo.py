"""§4.3 standard on-policy GRPO.

Suggested hyperparameters:
  n_train_examples = 6400, n_val_examples = 1024, num_rollout_steps = 200
  learning_rate = 1e-5, rollout_batch_size = 256, group_size = 8
  gradient_accumulation_steps = 32, max_tokens = 512, max_grad_norm = 1.0

Variants and off-policy live in scripts/train_grpo_variants.py.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from cs336_alignment.grpo_experiment import ROOT, add_common_args, run_training
from cs336_alignment.training_runtime import RECIPE_CONFIGS, TrainAlgorithmConfig


def standard_config(args: argparse.Namespace) -> TrainAlgorithmConfig:
    # Preset leaves batch sizes empty; the CLI supplies the §4.3 defaults.
    return replace(
        RECIPE_CONFIGS["standard"],
        rollout_batch_size=args.rollout_batch_size,
        train_batch_size=args.rollout_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="§4.3 standard on-policy GRPO")
    add_common_args(parser)
    args = parser.parse_args()
    args.recipe = "standard"
    if args.output_dir is None:
        args.output_dir = ROOT / "results" / f"standard_seed_{args.seed}"
    run_training(args, standard_config(args))


if __name__ == "__main__":
    main()
