import os
from typing import BinaryIO

import torch
from torch import nn
from torch.optim import Optimizer


CheckpointTarget = str | os.PathLike | BinaryIO


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: CheckpointTarget,
) -> None:
    """
    Save the model, optimizer, and current iteration.

    out can be:
    - a file path
    - a pathlib.Path
    - a binary file-like object
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    # torch.save itself accepts a path or a binary file-like object
    torch.save(checkpoint, out)


def load_checkpoint(
    src: CheckpointTarget,
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    """
    Load a checkpoint in place into an already constructed model and optimizer.

    Returns the iteration stored in the checkpoint.
    """

    # Load the full checkpoint dictionary first
    checkpoint = torch.load(
        src,
        map_location="cpu",
        weights_only=True,
    )

    # Restore model parameters in place
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Restore AdamW m, v, step, and param_groups in place
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # Must return the iteration
    return int(checkpoint["iteration"])
