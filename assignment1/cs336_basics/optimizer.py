import math
from collections.abc import Iterable

import torch
from torch import nn


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    根据当前迭代次数计算学习率。
    """

    # 第一阶段：线性 warmup
    if it <= warmup_iters:
        # 没有 warmup 时，直接从最大学习率开始
        if warmup_iters == 0:
            return max_learning_rate

        return max_learning_rate * it / warmup_iters

    # 第二阶段：余弦退火
    if it <= cosine_cycle_iters:
        cosine_progress = (
            (it - warmup_iters)
            / (cosine_cycle_iters - warmup_iters)
        )

        return (
            min_learning_rate
            + 0.5
            * (max_learning_rate - min_learning_rate)
            * (
                1
                + math.cos(
                    math.pi * cosine_progress
                )
            )
        )

    # 第三阶段：保持最小学习率
    return min_learning_rate


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError("lr must be non-negative")

        if eps < 0:
            raise ValueError("eps must be non-negative")

        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        beta1, beta2 = betas

        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")

        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")

        # 每一个 param_group 都可以拥有自己的一套超参数
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行一次 AdamW 参数更新。
        """

        loss = None

        # 支持可选的 closure，虽然本次测试通常用不到
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历不同的参数组
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                # 没有梯度的参数直接跳过
                if parameter.grad is None:
                    continue

                grad = parameter.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

                # 读取当前参数自己的优化器状态
                state = self.state[parameter]

                # 第一次遇到这个参数时初始化状态
                if len(state) == 0:
                    state["step"] = 0

                    # 一阶动量 m
                    state["exp_avg"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )

                    # 二阶动量 v
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # 当前参数的更新次数从 1 开始
                state["step"] += 1
                step = state["step"]

                # 1. 更新一阶动量
                # m = beta1 * m + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(
                    grad,
                    alpha=1 - beta1,
                )

                # 2. 更新二阶动量
                # v = beta2 * v + (1 - beta2) * grad^2
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                # 3. 计算偏差修正
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # 4. 解耦权重衰减
                # p = p - lr * weight_decay * p
                parameter.mul_(
                    1 - lr * weight_decay
                )

                # 5. Adam 参数更新
                #
                # m_hat = m / bias_correction1
                # sqrt(v_hat) = sqrt(v) / sqrt(bias_correction2)
                step_size = lr / bias_correction1

                denominator = (
                    exp_avg_sq.sqrt()
                    / math.sqrt(bias_correction2)
                ).add_(eps)

                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-step_size,
                )

        return loss