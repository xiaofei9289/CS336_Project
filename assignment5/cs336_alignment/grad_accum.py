"""Gradient-accumulation helpers shared by SFT and DPO."""

from __future__ import annotations

import torch


def apply_partial_window_scale(model: torch.nn.Module, accumulation_steps: int, pending: int) -> None:
    """Grads were divided by the full window. Rescale a short tail to its own mean."""
    if pending == accumulation_steps:
        return
    scale = accumulation_steps / pending
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(scale)


def average_sequence_grads(model: torch.nn.Module, num_sequences: int) -> None:
    """Turn a sum of per-sequence gradients into their mean."""
    for param in model.parameters():
        if param.grad is not None:
            param.grad.div_(num_sequences)
