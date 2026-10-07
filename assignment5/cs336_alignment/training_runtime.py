"""GRPO training runtime: recipe configs, group-aware batching, off-policy old_log_probs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import AutoTokenizer

from cs336_alignment.grpo import get_response_log_probs, grpo_train_step, tokenize_prompt_and_output
from cs336_alignment.rollout_batching import assert_batches_preserve_prompt_groups, build_train_batch_indices

RecipeName = Literal[
    "standard",
    "grpo_constant",
    "dr_grpo",
    "rft",
    "maxrl",
    "offpolicy_naive",
    "offpolicy_noclip",
    "offpolicy_clip",
    "offpolicy_gspo",
    "try_your_own",
]


@dataclass(frozen=True)
class TrainAlgorithmConfig:
    baseline: Literal["mean", "none"] = "mean"
    advantage_normalizer: Literal["std", "none", "mean"] = "std"
    loss_normalization: Literal["sequence", "constant"] = "sequence"
    normalization_constant: int | None = None
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none"
    cliprange: float | None = None
    rollout_batch_size: int | None = None
    train_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    off_policy_updates: int = 1


RECIPE_CONFIGS: dict[RecipeName, TrainAlgorithmConfig] = {
    "standard": TrainAlgorithmConfig(),
    "grpo_constant": TrainAlgorithmConfig(loss_normalization="constant"),
    "dr_grpo": TrainAlgorithmConfig(
        advantage_normalizer="none",
        loss_normalization="constant",
    ),
    "rft": TrainAlgorithmConfig(
        baseline="none",
        advantage_normalizer="none",
        loss_normalization="constant",
    ),
    "maxrl": TrainAlgorithmConfig(
        advantage_normalizer="mean",
        loss_normalization="constant",
    ),
    "offpolicy_naive": TrainAlgorithmConfig(
        importance_reweighting_method="none",
        train_batch_size=8,
        gradient_accumulation_steps=1,
        off_policy_updates=32,
    ),
    "offpolicy_noclip": TrainAlgorithmConfig(
        importance_reweighting_method="noclip",
        train_batch_size=8,
        gradient_accumulation_steps=1,
        off_policy_updates=32,
    ),
    "offpolicy_clip": TrainAlgorithmConfig(
        importance_reweighting_method="grpo",
        cliprange=0.2,
        train_batch_size=8,
        gradient_accumulation_steps=1,
        off_policy_updates=32,
    ),
    "offpolicy_gspo": TrainAlgorithmConfig(
        importance_reweighting_method="gspo",
        cliprange=3e-4,
        train_batch_size=8,
        gradient_accumulation_steps=1,
        off_policy_updates=32,
    ),
    "try_your_own": TrainAlgorithmConfig(
        advantage_normalizer="none",
        loss_normalization="sequence",
    ),
}


def validate_algorithm_config(config: TrainAlgorithmConfig, rollout_batch_size: int) -> None:
    train_batch_size = config.train_batch_size or rollout_batch_size
    grad_accum = config.gradient_accumulation_steps or 1
    if rollout_batch_size % train_batch_size != 0:
        raise ValueError("rollout_batch_size must be divisible by train_batch_size")
    updates = rollout_batch_size // train_batch_size
    if config.off_policy_updates != 1 and config.off_policy_updates != updates:
        raise ValueError(
            f"off_policy_updates={config.off_policy_updates} but "
            f"rollout_batch_size/train_batch_size={updates}"
        )
    if train_batch_size % grad_accum != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps")
    if config.importance_reweighting_method in ("grpo", "gspo") and config.cliprange is None:
        raise ValueError("cliprange is required for clipped off-policy methods")
    if config.loss_normalization == "constant" and config.normalization_constant is None:
        raise ValueError("normalization_constant is required for constant loss normalization")


@torch.no_grad()
def compute_old_log_probs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    responses: list[str],
) -> torch.Tensor:
    device = next(model.parameters()).device
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    scored = get_response_log_probs(
        model,
        tokenized["input_ids"].to(device),
        tokenized["labels"].to(device),
    )
    return scored["log_probs"].detach().cpu()


def run_train_updates(
    policy: torch.nn.Module,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    reward_fn,
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    config: TrainAlgorithmConfig,
    group_size: int,
    max_grad_norm: float,
    rng: random.Random,
    compute_old_log_probs_for_offpolicy: bool,
) -> tuple[float, dict[str, float | bool]]:
    rollout_batch_size = len(prompts)
    train_batch_size = config.train_batch_size or rollout_batch_size
    grad_accum = config.gradient_accumulation_steps or 1
    num_updates = config.off_policy_updates if config.off_policy_updates > 1 else 1

    batches = build_train_batch_indices(
        rollout_batch_size=rollout_batch_size,
        group_size=group_size,
        train_batch_size=train_batch_size,
        num_updates=num_updates,
        rng=rng,
    )
    assert_batches_preserve_prompt_groups(prompts, batches, group_size)

    precomputed_old: list[torch.Tensor] = []
    if compute_old_log_probs_for_offpolicy:
        was_training = policy.training
        policy.eval()
        try:
            for batch_idx in batches:
                batch_prompts = [prompts[i] for i in batch_idx]
                batch_responses = [responses[i] for i in batch_idx]
                precomputed_old.append(
                    compute_old_log_probs(policy, tokenizer, batch_prompts, batch_responses)
                )
        finally:
            policy.train(was_training)

    total_loss = 0.0
    agg: dict[str, list[float]] = {}

    for update_idx, batch_idx in enumerate(batches):
        batch_prompts = [prompts[i] for i in batch_idx]
        batch_responses = [responses[i] for i in batch_idx]
        batch_ground_truths = [ground_truths[i] for i in batch_idx]
        batch_old = precomputed_old[update_idx] if precomputed_old else None

        loss, metadata = grpo_train_step(
            model=policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=grad_accum,
            max_grad_norm=max_grad_norm,
            reward_fn=reward_fn,
            repeated_prompts=batch_prompts,
            rollout_responses=batch_responses,
            repeated_ground_truths=batch_ground_truths,
            group_size=group_size,
            baseline=config.baseline,
            advantage_normalizer=config.advantage_normalizer,
            importance_reweighting_method=config.importance_reweighting_method,
            old_log_probs=batch_old,
            cliprange=config.cliprange,
            loss_normalization=config.loss_normalization,
            normalization_constant=config.normalization_constant,
        )
        total_loss += float(loss.detach())
        for key, value in metadata.items():
            if isinstance(value, torch.Tensor):
                agg.setdefault(key, []).append(float(value.detach()))
            elif isinstance(value, bool):
                agg.setdefault(key, []).append(value)
            else:
                agg.setdefault(key, []).append(float(value))

    mean_metadata: dict[str, float | bool] = {}
    for key, values in agg.items():
        if key == "token_entropy_skipped":
            mean_metadata[key] = all(values)
        else:
            mean_metadata[key] = sum(values) / len(values)
    return total_loss / num_updates, mean_metadata
