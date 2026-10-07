"""CPU integration: group batching, old_log_probs per minibatch, run_train_updates."""

from __future__ import annotations

import random

import torch

from cs336_alignment.grpo import get_response_log_probs, tokenize_prompt_and_output
from cs336_alignment.rollout_batching import build_train_batch_indices
from cs336_alignment.training_runtime import TrainAlgorithmConfig, compute_old_log_probs, run_train_updates


def _train_step_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
    del ground_truth
    reward = float(len(response) % 3) / 2.0
    return {
        "reward": reward,
        "format_reward": float(reward > 0),
        "answer_reward": reward,
    }


def test_run_train_updates_off_policy_old_log_probs_shape(tiny_train_model, tokenizer):
    group_size = 2
    rollout_batch_size = 16
    num_groups = rollout_batch_size // group_size
    prompts = []
    responses = []
    ground_truths = []
    for group_id in range(num_groups):
        for _ in range(group_size):
            prompts.append(f"Hello group {group_id}")
            responses.append(f"resp {group_id} {len(prompts)}")
            ground_truths.append("42")

    config = TrainAlgorithmConfig(
        importance_reweighting_method="noclip",
        cliprange=0.1,
        train_batch_size=4,
        gradient_accumulation_steps=1,
        off_policy_updates=4,
    )
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=1e-3)
    rng = random.Random(0)

    loss, metadata = run_train_updates(
        policy=tiny_train_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        reward_fn=_train_step_reward_fn,
        prompts=prompts,
        responses=responses,
        ground_truths=ground_truths,
        config=config,
        group_size=group_size,
        max_grad_norm=1.0,
        rng=rng,
        compute_old_log_probs_for_offpolicy=True,
    )
    assert isinstance(loss, float)
    assert "mean_reward" in metadata

    batches = build_train_batch_indices(
        rollout_batch_size=rollout_batch_size,
        group_size=group_size,
        train_batch_size=4,
        num_updates=4,
        rng=random.Random(0),
    )
    batch_idx = batches[0]
    batch_prompts = [prompts[i] for i in batch_idx]
    batch_responses = [responses[i] for i in batch_idx]
    old_log_probs = compute_old_log_probs(
        tiny_train_model, tokenizer, batch_prompts, batch_responses
    )
    tokenized = tokenize_prompt_and_output(batch_prompts, batch_responses, tokenizer)
    scored = get_response_log_probs(
        tiny_train_model,
        tokenized["input_ids"],
        tokenized["labels"],
    )
    assert old_log_probs.shape == scored["log_probs"].shape
    assert torch.allclose(old_log_probs, scored["log_probs"].detach().cpu())
