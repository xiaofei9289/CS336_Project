"""Rollout batching: keep each group_size prompt block contiguous; shuffle groups only."""

from __future__ import annotations

import random


def build_train_batch_indices(
    rollout_batch_size: int,
    group_size: int,
    train_batch_size: int,
    num_updates: int,
    rng: random.Random,
) -> list[list[int]]:
    """Return num_updates train batches, each with train_batch_size rollout indices.

    Invariant: indices in [g*group_size, (g+1)*group_size) share the same prompt under repeat_group.
    For multi-step off-policy, only shuffle prompt groups, not within-group order.
    """
    if rollout_batch_size % group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")
    if train_batch_size % group_size != 0:
        raise ValueError("train_batch_size must be divisible by group_size")

    num_groups = rollout_batch_size // group_size
    groups_per_update = train_batch_size // group_size
    if num_updates * groups_per_update != num_groups:
        raise ValueError(
            f"num_updates * (train_batch_size/group_size) must equal num_groups, "
            f"got {num_updates} * {groups_per_update} != {num_groups}"
        )

    group_ids = list(range(num_groups))
    if num_updates > 1:
        rng.shuffle(group_ids)

    batches: list[list[int]] = []
    for update_idx in range(num_updates):
        batch_indices: list[int] = []
        chunk = group_ids[update_idx * groups_per_update : (update_idx + 1) * groups_per_update]
        for group_id in chunk:
            start = group_id * group_size
            batch_indices.extend(range(start, start + group_size))
        batches.append(batch_indices)
    return batches


def assert_batches_preserve_prompt_groups(
    prompts: list[str],
    batches: list[list[int]],
    group_size: int,
) -> None:
    """Within each batch, every group_size window must share the same prompt."""
    for batch_idx in batches:
        for offset in range(0, len(batch_idx), group_size):
            window = batch_idx[offset : offset + group_size]
            if len(window) != group_size:
                raise AssertionError(f"incomplete group window: {window}")
            prompt_set = {prompts[i] for i in window}
            if len(prompt_set) != 1:
                raise AssertionError(
                    f"mixed prompts in one group: {prompt_set} indices={window}"
                )
