"""
§4.2 on-policy GRPO. Public signatures match the handout.

§5 baselines, constant normalization, and zero-advantage pruning: grpo_variants.py
§6 importance ratios and clipping: grpo_offpolicy.py
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
from einops import rearrange
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer

from cs336_alignment.grpo_offpolicy import (
    accumulate_clip_fraction,
    attach_clip_fraction,
    offpolicy_per_token_loss,
    slice_old_log_probs,
)
from cs336_alignment.grpo_variants import (
    aggregate_constant_loss,
    nonzero_advantage_indices,
    rescale_pruned_sequence_loss,
    variant_normalizer,
)


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt and output separately, concat, return input_ids, labels, response_mask.

    No extra special tokens between prompt and response.
    input_ids drops the last token; labels drop the first token.
    response_mask aligns with labels: 1 on response tokens, 0 on prompt/padding.
    Shapes are (batch_size, max_len - 1).
    """
    prompt_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs
    ]
    output_ids = [
        tokenizer.encode(output, add_special_tokens=False) for output in output_strs
    ]
    sequences = [prompt + output for prompt, output in zip(prompt_ids, output_ids)]
    max_len = max(len(sequence) for sequence in sequences)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id for padding")

    input_rows = []
    label_rows = []
    mask_rows = []
    for prompt, sequence in zip(prompt_ids, sequences):
        padded = sequence + [pad_id] * (max_len - len(sequence))
        prompt_len = len(prompt)
        response_end = len(sequence)
        input_rows.append(padded[:-1])
        label_rows.append(padded[1:])
        mask_rows.append(
            [
                prompt_len <= index < response_end
                for index in range(1, max_len)
            ]
        )

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "response_mask": torch.tensor(mask_rows, dtype=torch.bool),
    }


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Per-token log p_theta(x_t | x_<t); optional next-token entropy per position.

    log_probs matches labels shape. entropy appears only if return_token_entropy=True.
    """
    logits = model(input_ids).logits
    # BF16 log-softmax cannot resolve GSPO's 3e-4 clip. Score in FP32.
    log_probs_full = torch.log_softmax(logits.float(), dim=-1)
    log_probs = rearrange(
        torch.gather(log_probs_full, 2, rearrange(labels, "b s -> b s 1")),
        "b s 1 -> b s",
    )
    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = -(log_probs_full.exp() * log_probs_full).sum(dim=-1)
    return result


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Score each rollout with reward_fn.

    raw_rewards shape (rollout_batch_size,).
    metadata includes at least mean total and mean format reward over the batch.
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            "rollout_responses and repeated_ground_truths must have the same length, "
            f"got {len(rollout_responses)} and {len(repeated_ground_truths)}"
        )
    scores = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(
        [score["reward"] for score in scores], dtype=torch.float32
    )
    format_rewards = torch.tensor(
        [score["format_reward"] for score in scores], dtype=torch.float32
    )
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": format_rewards.mean().item(),
        "mean_answer_reward": float(
            sum(score["answer_reward"] for score in scores) / len(scores)
        ),
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Group flat rewards by group_size. Standard GRPO subtracts the group mean and divides by std."""
    grouped = rearrange(raw_rewards, "(g n) -> g n", n=group_size).float()
    group_mean = grouped.mean(dim=-1, keepdim=True)
    if baseline == "mean":
        advantages = grouped - group_mean
    elif baseline == "none":
        advantages = grouped
    else:
        raise NotImplementedError
    if advantage_normalizer == "std":
        scale = grouped.std(dim=-1, keepdim=True, unbiased=True) + advantage_eps
        advantages = advantages / scale
    else:
        advantages = variant_normalizer(
            advantages,
            group_mean,
            advantage_normalizer,
            advantage_eps,
        )

    metadata = {
        "reward_mean": grouped.mean().item(),
        "reward_std": grouped.std(unbiased=False).item(),
        "reward_max": grouped.max().item(),
        "reward_min": grouped.min().item(),
    }
    return rearrange(advantages, "g n -> (g n)"), metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-token on-policy loss, negated for gradient ascent. Other methods go to grpo_offpolicy."""
    advantages = raw_rewards_or_advantages
    if advantages.ndim == 1:
        advantages = rearrange(advantages, "b -> b 1")

    if importance_reweighting_method == "none":
        return -(advantages * policy_log_probs), {}

    return offpolicy_per_token_loss(
        advantages,
        policy_log_probs,
        importance_reweighting_method,
        old_log_probs,
        cliprange,
        response_mask,
    )


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Sequence normalization: mean over masked tokens, then mean over sequences."""
    if loss_normalization == "constant":
        return aggregate_constant_loss(
            per_token_policy_gradient_loss,
            mask,
            normalization_constant,
        )
    if loss_normalization != "sequence":
        raise ValueError(f"Unknown loss normalization: {loss_normalization}")
    mask_f = mask.to(dtype=per_token_policy_gradient_loss.dtype)
    token_count = mask_f.sum(dim=-1).clamp(min=1)
    per_sequence = (per_token_policy_gradient_loss * mask_f).sum(dim=-1) / token_count
    return per_sequence.mean()


def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """One standard on-policy update: rewards, advantages, microbatch fwd/bwd, clip, step.

    Sequence normalization scales each microbatch loss by microbatch_size / batch_size.
    """
    device = next(model.parameters()).device
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )
    advantages, _ = compute_group_normalized_rewards(
        raw_rewards.to(device),
        group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    batch_size = len(repeated_prompts)
    if gradient_accumulation_steps < 1 or batch_size % gradient_accumulation_steps != 0:
        raise ValueError("batch_size must be divisible by gradient_accumulation_steps")
    microbatch_size = batch_size // gradient_accumulation_steps
    logged_loss = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    entropy_count = torch.zeros((), device=device)
    clip_fraction_sum = torch.zeros((), device=device)
    clip_fraction_count = torch.zeros((), device=device)

    optimizer.zero_grad(set_to_none=True)
    had_active_sequences = False
    for start in range(0, batch_size, microbatch_size):
        active_idx = nonzero_advantage_indices(advantages[start : start + microbatch_size], start)
        if active_idx is None:
            continue
        had_active_sequences = True
        kept = active_idx.detach().cpu().tolist()
        tokenized = tokenize_prompt_and_output(
            [repeated_prompts[i] for i in kept],
            [rollout_responses[i] for i in kept],
            tokenizer,
        )
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)
        scored = get_response_log_probs(
            model,
            input_ids,
            labels,
            return_token_entropy=True,
        )
        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            advantages[active_idx],
            scored["log_probs"],
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=slice_old_log_probs(
                old_log_probs, active_idx, device, input_ids.shape[1]
            ),
            cliprange=cliprange,
            response_mask=response_mask,
        )
        loss = aggregate_loss_across_microbatch(
            per_token_loss,
            response_mask,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )
        if loss_normalization == "sequence":
            loss = rescale_pruned_sequence_loss(loss, active_idx.shape[0], microbatch_size)
            scaled_loss = loss * (microbatch_size / batch_size)
        else:
            scaled_loss = loss
        logged_loss = logged_loss + scaled_loss.detach()
        scaled_loss.backward()

        token_entropy = scored["token_entropy"].detach()
        entropy_sum = entropy_sum + token_entropy[response_mask].sum()
        entropy_count = entropy_count + response_mask.sum()
        clip_fraction_sum, clip_fraction_count = accumulate_clip_fraction(
            loss_metadata, clip_fraction_sum, clip_fraction_count
        )

    if not had_active_sequences:
        # No backward ran, so grads are None. A zero tensor lets Adam decay momentum,
        # matching an unpruned step whose loss is exactly zero.
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.zeros_like(param)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm if max_grad_norm is not None else float("inf"),
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metadata = {
        "loss": logged_loss.detach(),
        "grad_norm": grad_norm.detach(),
        "token_entropy": (entropy_sum / entropy_count.clamp(min=1)).detach(),
        "token_entropy_skipped": bool(entropy_count.item() == 0),
        "mean_reward": reward_metadata["mean_reward"],
        "mean_format_reward": reward_metadata["mean_format_reward"],
        "mean_answer_reward": reward_metadata["mean_answer_reward"],
    }
    return logged_loss, attach_clip_fraction(
        metadata,
        importance_reweighting_method,
        clip_fraction_sum,
        clip_fraction_count,
        device,
        skipped=not had_active_sequences,
    )
