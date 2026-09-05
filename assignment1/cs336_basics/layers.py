import torch
import torch.nn as nn
import math
from torch import Tensor
from jaxtyping import Float, Int


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # 每一行对应一个 token 的向量
        # shape: (vocab_size, d_model)
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )

        # Embedding 初始化：
        # N(0, 1)，截断到 [-3, 3]
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(
        self,
        token_ids: Int[Tensor, "..."],
    ) -> Float[Tensor, "... d_model"]:
        # token_ids 中的每个整数都作为 weight 的行索引
        return self.weight[token_ids]


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        # σ = sqrt(2 / (d_in + d_out))
        sigma = math.sqrt(
            2.0 / (in_features + out_features)
        )

        # 从 N(0, σ²) 中采样，并截断到 [-3σ, 3σ]
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )

    def forward(self, x):
        return x @ self.weight.T



class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # gain 参数，形状为 (d_model,)
        # RMSNorm 的 gain 初始化为 1
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(
        self,
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:

        # 保存原始 dtype，例如 BF16
        input_dtype = x.dtype

        # 为提高数值稳定性，在 FP32 中计算归一化
        x_float = x.to(torch.float32)

        # 对最后一维计算：
        # 1 / sqrt(mean(x²) + eps)
        inverse_rms = torch.rsqrt(
            x_float.pow(2).mean(
                dim=-1,
                keepdim=True,
            )
            + self.eps
        )

        # (..., d_model) * (..., 1)
        normalized = x_float * inverse_rms

        # weight: (d_model,)
        # 自动广播到所有前置维度
        output = normalized * self.weight

        # 转回输入的原始 dtype
        return output.to(input_dtype)

def silu(
    in_features: Float[Tensor, "..."],
) -> Float[Tensor, "..."]:
    """
    Apply the SiLU activation element-wise.

    SiLU(x) = x * sigmoid(x)
    """
    return in_features * torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # (d_model) -> (d_ff)
        self.w1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

        # (d_ff) -> (d_model)
        self.w2 = Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        # (d_model) -> (d_ff)
        self.w3 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:
        """
        SwiGLU(x) = W2(SiLU(W1x) ⊙ W3x)
        """

        gate = silu(self.w1(x))
        value = self.w3(x)

        return self.w2(gate * value)


def softmax(
    in_features: Float[Tensor, "..."],
    dim: int,
    ) -> Float[Tensor, "..."]:
    """
    在指定的 dim 维度上计算数值稳定的 Softmax。

    Softmax(x_i) = exp(x_i - max(x))
                   -----------------
                   sum(exp(x_j - max(x)))
    """

    # 1. 在指定维度上找到最大值
    # keepdim=True 保留这一维，方便后续广播
    max_value = torch.max(
        in_features,
        dim=dim,
        keepdim=True,
    ).values

    # 2. 所有元素先减去最大值，防止 exp 溢出
    shifted_features = in_features - max_value

    # 3. 对平移后的数计算指数
    exp_features = torch.exp(shifted_features)

    # 4. 在同一个维度上计算指数和
    exp_sum = torch.sum(
        exp_features,
        dim=dim,
        keepdim=True,
    )

    # 5. 每个指数除以指数总和
    return exp_features / exp_sum

#--------------------------------


def cross_entropy(
    inputs: Float[Tensor, "batch vocab_size"],
    targets: Int[Tensor, "batch"],
) -> Float[Tensor, ""]:
    """
    计算多分类交叉熵损失。

    inputs:
        未经过 softmax 的 logits
        形状为 (batch, vocab_size)

    targets:
        每个样本的正确类别下标
        形状为 (batch,)

    return:
        batch 中所有样本损失的平均值，是一个标量
    """

    # 1. 每一行减去该行的最大值，防止 exp 溢出
    max_logits = inputs.max(
        dim=-1,
        keepdim=True,
    ).values

    shifted_logits = inputs - max_logits

    # 2. 取出每个样本对应正确类别的 logit
    #
    # targets.unsqueeze(-1):
    # (batch,) -> (batch, 1)
    #
    # gather 后：
    # (batch, 1) -> squeeze -> (batch,)
    target_logits = shifted_logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # 3. 计算每一行的 log(sum(exp(logits)))
    log_sum_exp = torch.log(
        torch.exp(shifted_logits).sum(dim=-1)
    )

    # 4. 每个样本的交叉熵
    losses = -target_logits + log_sum_exp

    # 5. 对 batch 取平均，返回标量
    return losses.mean()

#--------------------------------
from collections.abc import Iterable

import torch
from torch import nn


def gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
) -> None:
    """
    按所有参数梯度的全局 L2 范数进行裁剪。

    parameters:
        模型参数

    max_l2_norm:
        允许的最大全局梯度 L2 范数
    """

    # 1. 只保留存在梯度的参数
    grads = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]

    # 没有任何梯度时，不需要处理
    if len(grads) == 0:
        return

    # 2. 计算所有梯度合在一起的全局 L2 范数
    #
    # total_norm =
    # sqrt(
    #     sum(
    #         sum(grad ** 2)
    #     )
    # )
    total_squared_norm = torch.stack([
        grad.detach().pow(2).sum()
        for grad in grads
    ]).sum()

    total_norm = torch.sqrt(total_squared_norm)

    # 3. 总范数没有超过上限时，保持梯度不变
    if total_norm <= max_l2_norm:
        return

    # 4. 所有梯度共用同一个缩放系数
    scale = max_l2_norm / (total_norm + 1e-6)

    # 5. 原地修改每个参数原有的 grad
    with torch.no_grad():
        for grad in grads:
            grad.mul_(scale)