import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Randomly sample next-token training examples from a 1D token sequence.

    Returns:
        x: (batch_size, context_length)
        y: (batch_size, context_length)
    """

    dataset_length = len(dataset)

    # Each example needs context_length + 1 tokens
    if dataset_length < context_length + 1:
        raise ValueError(
            "dataset must contain at least "
            "context_length + 1 tokens"
        )

    # Maximum valid start index:
    # start + context_length <= dataset_length - 1
    max_start = dataset_length - context_length - 1

    # np.random.randint treats high as exclusive,
    # so pass max_start + 1
    start_indices = np.random.randint(
        low=0,
        high=max_start + 1,
        size=batch_size,
    )

    # Slice x from each start, and y shifted one token to the right
    x_array = np.stack([
        dataset[start : start + context_length]
        for start in start_indices
    ])

    y_array = np.stack([
        dataset[start + 1 : start + context_length + 1]
        for start in start_indices
    ])

    # Convert to LongTensor and move to the requested device
    x = torch.tensor(
        x_array,
        dtype=torch.long,
        device=device,
    )

    y = torch.tensor(
        y_array,
        dtype=torch.long,
        device=device,
    )

    return x, y
