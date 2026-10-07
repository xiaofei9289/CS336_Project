"""§5 on-policy variants: baselines, constant loss, and zero-advantage sequences."""

from __future__ import annotations

from typing import Literal

import torch


def variant_normalizer(
    advantages: torch.Tensor,
    group_mean: torch.Tensor,
    advantage_normalizer: Literal["std", "none", "mean"],
    advantage_eps: float,
) -> torch.Tensor:
    """Normalizers other than per-group standard deviation."""
    if advantage_normalizer == "mean":
        return advantages / (group_mean + advantage_eps)
    if advantage_normalizer == "none":
        return advantages
    raise NotImplementedError


def aggregate_constant_loss(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    normalization_constant: int | None,
) -> torch.Tensor:
    if normalization_constant is None:
        raise ValueError("normalization_constant is required for constant loss")
    mask_f = mask.to(dtype=per_token_policy_gradient_loss.dtype)
    return (per_token_policy_gradient_loss * mask_f).sum() / normalization_constant


def nonzero_advantage_indices(chunk: torch.Tensor, start: int) -> torch.Tensor | None:
    """Absolute batch indices of sequences with nonzero advantage, or None."""
    keep = chunk != 0
    if not bool(keep.any()):
        return None
    return torch.arange(start, start + chunk.shape[0], device=chunk.device)[keep]


def rescale_pruned_sequence_loss(
    loss: torch.Tensor,
    num_active: int,
    microbatch_size: int,
) -> torch.Tensor:
    """Make a mean over kept sequences match a mean over the original microbatch."""
    return loss * (num_active / microbatch_size)

