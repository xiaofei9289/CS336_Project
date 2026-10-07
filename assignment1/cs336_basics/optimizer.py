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
    Compute the learning rate for the current iteration.
    """

    # Stage 1: linear warmup
    if it <= warmup_iters:
        # With no warmup, start directly at the maximum learning rate
        if warmup_iters == 0:
            return max_learning_rate

        return max_learning_rate * it / warmup_iters

    # Stage 2: cosine annealing
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

    # Stage 3: hold the minimum learning rate
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

        # Each param group can have its own set of hyperparameters
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
        Perform one AdamW parameter update.
        """

        loss = None

        # Support an optional closure, although these tests usually do not use it
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate over parameter groups
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                # Skip parameters that have no gradient
                if parameter.grad is None:
                    continue

                grad = parameter.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

                # Read this parameter's optimizer state
                state = self.state[parameter]

                # Initialize state the first time this parameter is seen
                if len(state) == 0:
                    state["step"] = 0

                    # First moment m
                    state["exp_avg"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )

                    # Second moment v
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # This parameter's update count starts at 1
                state["step"] += 1
                step = state["step"]

                # 1. Update the first moment
                # m = beta1 * m + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(
                    grad,
                    alpha=1 - beta1,
                )

                # 2. Update the second moment
                # v = beta2 * v + (1 - beta2) * grad^2
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                # 3. Compute bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # 4. Decoupled weight decay
                # p = p - lr * weight_decay * p
                parameter.mul_(
                    1 - lr * weight_decay
                )

                # 5. Adam parameter update
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
