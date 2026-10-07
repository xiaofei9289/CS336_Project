"""§5.4 on-policy variants, §6.4 off-policy, and §7 try-your-own.

Off-policy (§6.4): rollout_batch_size=256, train_batch_size=8,
  off_policy_updates=32, gradient_accumulation_steps=1.
"""

from __future__ import annotations

import argparse

from cs336_alignment.grpo_experiment import ROOT, add_common_args, run_training
from cs336_alignment.training_runtime import RECIPE_CONFIGS, TrainAlgorithmConfig


def resolve_algorithm_config(args: argparse.Namespace) -> TrainAlgorithmConfig:
    base = RECIPE_CONFIGS[args.recipe]
    rollout_batch_size = args.rollout_batch_size
    train_batch_size = base.train_batch_size or rollout_batch_size
    grad_accum = (
        base.gradient_accumulation_steps
        if base.gradient_accumulation_steps is not None
        else args.gradient_accumulation_steps
    )
    off_policy_updates = base.off_policy_updates
    if args.off_policy_updates is not None:
        off_policy_updates = args.off_policy_updates
    if args.train_batch_size is not None:
        train_batch_size = args.train_batch_size

    norm_constant = args.normalization_constant
    if norm_constant is None and base.loss_normalization == "constant":
        # Handout §5.1 Dr. GRPO: Z = B * G * L = rollout_batch_size * max_tokens
        norm_constant = rollout_batch_size * args.max_tokens

    cliprange = args.cliprange if args.cliprange is not None else base.cliprange
    baseline = args.baseline if args.baseline is not None else base.baseline
    advantage_normalizer = (
        args.advantage_normalizer if args.advantage_normalizer is not None else base.advantage_normalizer
    )
    loss_normalization = (
        args.loss_normalization if args.loss_normalization is not None else base.loss_normalization
    )
    importance = (
        args.importance_reweighting_method
        if args.importance_reweighting_method is not None
        else base.importance_reweighting_method
    )

    return TrainAlgorithmConfig(
        baseline=baseline,
        advantage_normalizer=advantage_normalizer,
        loss_normalization=loss_normalization,
        normalization_constant=norm_constant,
        importance_reweighting_method=importance,
        cliprange=cliprange,
        rollout_batch_size=rollout_batch_size,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=grad_accum,
        off_policy_updates=off_policy_updates,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO variants and off-policy training")
    add_common_args(parser)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="off-policy: responses per optimizer update",
    )
    parser.add_argument(
        "--off-policy-updates",
        type=int,
        default=None,
        help="training steps per rollout batch",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        choices=[name for name in RECIPE_CONFIGS if name != "standard"],
        required=True,
        help="grpo_constant / dr_grpo / rft / maxrl / offpolicy_* / try_your_own",
    )
    parser.add_argument("--baseline", choices=["mean", "none"], default=None)
    parser.add_argument("--advantage-normalizer", choices=["std", "none", "mean"], default=None)
    parser.add_argument("--loss-normalization", choices=["sequence", "constant"], default=None)
    parser.add_argument("--normalization-constant", type=int, default=None)
    parser.add_argument(
        "--importance-reweighting-method",
        choices=["none", "noclip", "grpo", "gspo"],
        default=None,
    )
    parser.add_argument("--cliprange", type=float, default=None)

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = ROOT / "results" / f"{args.recipe}_seed_{args.seed}"
    run_training(args, resolve_algorithm_config(args))


if __name__ == "__main__":
    main()
