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
    从一维 token 序列中随机抽取 next-token 训练数据。

    返回：
        x: (batch_size, context_length)
        y: (batch_size, context_length)
    """

    dataset_length = len(dataset)

    # 每条样本需要 context_length + 1 个 token
    if dataset_length < context_length + 1:
        raise ValueError(
            "dataset must contain at least "
            "context_length + 1 tokens"
        )

    # 最大合法起点：
    # start + context_length <= dataset_length - 1
    max_start = dataset_length - context_length - 1

    # np.random.randint 的 high 是开区间，
    # 所以传入 max_start + 1
    start_indices = np.random.randint(
        low=0,
        high=max_start + 1,
        size=batch_size,
    )

    # 从每个起点截取 x 和右移一位后的 y
    x_array = np.stack([
        dataset[start : start + context_length]
        for start in start_indices
    ])

    y_array = np.stack([
        dataset[start + 1 : start + context_length + 1]
        for start in start_indices
    ])

    # 转换成 LongTensor，并移动到指定设备
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