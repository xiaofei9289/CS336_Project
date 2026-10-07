"""§6 off-policy losses: noclip, token-level GRPO clip, and GSPO sequence clip."""

from __future__ import annotations

from typing import Literal

import torch


def slice_old_log_probs(
    old_log_probs: torch.Tensor | None,
    active_idx: torch.Tensor,
    device: torch.device,
    seq_len: int | None = None,
) -> torch.Tensor | None:
    if old_log_probs is None:
        return None
    # old_log_probs is stored on CPU. active_idx follows the policy device.
    index = active_idx.to(device=old_log_probs.device)
    selected = old_log_probs.index_select(0, index)
    # Right-padded rows: a shorter re-tokenization matches the prefix.
    if seq_len is not None:
        selected = selected[:, :seq_len]
    return selected.to(device)


def offpolicy_per_token_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["noclip", "grpo", "gspo"],
    old_log_probs: torch.Tensor | None,
    cliprange: float | None,
    response_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if old_log_probs is None:
        raise ValueError("old_log_probs is required for off-policy loss")
    # BF16 cannot represent GSPO's [1-3e-4, 1+3e-4]. Keep the ratio in FP32.
    advantages = advantages.float()
    log_ratio = policy_log_probs.float() - old_log_probs.float()

    if importance_reweighting_method == "gspo":
        if response_mask is None:
            weights = torch.ones_like(log_ratio)
        else:
            weights = response_mask.to(dtype=log_ratio.dtype)
        token_count = weights.sum(dim=-1, keepdim=True).clamp(min=1)
        sequence_ratio = torch.exp((log_ratio * weights).sum(dim=-1, keepdim=True) / token_count)
        ratio = sequence_ratio.expand_as(log_ratio)
    elif importance_reweighting_method in ("noclip", "grpo"):
        ratio = torch.exp(log_ratio)
    else:
        raise NotImplementedError

    if importance_reweighting_method == "noclip":
        return -(advantages * ratio), {}

    if cliprange is None:
        raise ValueError("cliprange is required for clipped off-policy loss")
    clipped_ratio = ratio.clamp(1 - cliprange, 1 + cliprange)
    objective = torch.minimum(advantages * ratio, advantages * clipped_ratio)
    clipped = (ratio != clipped_ratio).to(dtype=ratio.dtype)
    if response_mask is not None:
        mask_f = response_mask.to(dtype=ratio.dtype)
        clip_fraction = clipped.mul(mask_f).sum() / mask_f.sum().clamp(min=1)
    else:
        clip_fraction = clipped.mean()
    return -objective, {"clip_fraction": clip_fraction.detach()}


def accumulate_clip_fraction(
    loss_metadata: dict[str, torch.Tensor],
    clip_fraction_sum: torch.Tensor,
    clip_fraction_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "clip_fraction" not in loss_metadata:
        return clip_fraction_sum, clip_fraction_count
    return clip_fraction_sum + loss_metadata["clip_fraction"], clip_fraction_count + 1


def attach_clip_fraction(
    metadata: dict[str, torch.Tensor | float | bool],
    importance_reweighting_method: str,
    clip_fraction_sum: torch.Tensor,
    clip_fraction_count: torch.Tensor,
    device: torch.device,
    *,
    skipped: bool,
) -> dict[str, torch.Tensor | float | bool]:
    if skipped:
        if importance_reweighting_method not in ("none", "noclip"):
            metadata["clip_fraction"] = torch.zeros((), device=device)
        return metadata
    if clip_fraction_count.item() > 0:
        metadata["clip_fraction"] = (clip_fraction_sum / clip_fraction_count).detach()
    return metadata
