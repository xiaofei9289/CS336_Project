import random

import pytest

from cs336_alignment.rollout_batching import (
    assert_batches_preserve_prompt_groups,
    build_train_batch_indices,
)


def _prompts_for_rollout(rollout_batch_size: int, group_size: int) -> list[str]:
    num_groups = rollout_batch_size // group_size
    prompts = []
    for group_id in range(num_groups):
        prompts.extend([f"prompt-{group_id}"] * group_size)
    return prompts


def test_on_policy_single_batch_keeps_group_order():
    rng = random.Random(0)
    batches = build_train_batch_indices(
        rollout_batch_size=256,
        group_size=8,
        train_batch_size=256,
        num_updates=1,
        rng=rng,
    )
    assert len(batches) == 1
    assert batches[0] == list(range(256))
    prompts = _prompts_for_rollout(256, 8)
    assert_batches_preserve_prompt_groups(prompts, batches, 8)


def test_off_policy_shuffles_groups_not_within_group():
    rng = random.Random(42)
    batches = build_train_batch_indices(
        rollout_batch_size=256,
        group_size=8,
        train_batch_size=8,
        num_updates=32,
        rng=rng,
    )
    assert len(batches) == 32
    assert sum(len(batch) for batch in batches) == 256
    prompts = _prompts_for_rollout(256, 8)
    assert_batches_preserve_prompt_groups(prompts, batches, 8)
    flat = [idx for batch in batches for idx in batch]
    assert sorted(flat) == list(range(256))


def test_assert_rejects_mixed_group():
    prompts = ["a", "a", "b", "b"]
    with pytest.raises(AssertionError):
        assert_batches_preserve_prompt_groups(prompts, [[0, 2, 1, 3]], 2)
