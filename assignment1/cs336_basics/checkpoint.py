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
    保存模型、优化器和当前迭代次数。

    out 可以是：
    - 文件路径
    - pathlib.Path
    - 二进制 file-like 对象
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    # torch.save 本身支持路径和二进制 file-like
    torch.save(checkpoint, out)


def load_checkpoint(
    src: CheckpointTarget,
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    """
    将 checkpoint 原地加载进已经创建好的模型和优化器。

    返回保存时的迭代次数。
    """

    # 先加载完整的 checkpoint 字典
    checkpoint = torch.load(
        src,
        map_location="cpu",
        weights_only=True,
    )

    # 原地恢复模型参数
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # 原地恢复 AdamW 的 m、v、step 和 param_groups
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # 必须返回 iteration
    return int(checkpoint["iteration"])