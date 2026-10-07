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

        # Each row is one token's vector
        # shape: (vocab_size, d_model)
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )

        # Embedding initialization:
        # N(0, 1), truncated to [-3, 3]
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
        # Each integer in token_ids indexes a row of weight
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

        # Sample from N(0, σ²) and truncate to [-3σ, 3σ]
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

        # Gain parameter, shape (d_model,)
        # RMSNorm gain is initialized to 1
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

        # Save the original dtype, for example BF16
        input_dtype = x.dtype

        # Compute the normalization in FP32 for numerical stability
        x_float = x.to(torch.float32)

        # Over the last dimension:
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
        # Broadcasts automatically over all leading dimensions
        output = normalized * self.weight

        # Cast back to the input dtype
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

# ===== [added: SiLU FFN] start =====
class SiLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # Up-project: (..., d_model) -> (..., d_ff)
        # d_ff is passed in; it is not hard-coded to 2048.
        self.w1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

        # Down-project: (..., d_ff) -> (..., d_model)
        self.w2 = Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:
        # Up-project -> SiLU -> down-project
        # There is no w3 and no gating product between two branches.
        return self.w2(silu(self.w1(x)))
# ===== [added: SiLU FFN] end =====


def softmax(
    in_features: Float[Tensor, "..."],
    dim: int,
    ) -> Float[Tensor, "..."]:
    """
    Numerically stable softmax along the given dimension.

    Softmax(x_i) = exp(x_i - max(x))
                   -----------------
                   sum(exp(x_j - max(x)))
    """

    # 1. Max along the given dimension
    # keepdim=True keeps that dimension so later ops can broadcast
    max_value = torch.max(
        in_features,
        dim=dim,
        keepdim=True,
    ).values

    # 2. Subtract the max from every element to avoid exp overflow
    shifted_features = in_features - max_value

    # 3. Exponentiate the shifted values
    exp_features = torch.exp(shifted_features)

    # 4. Sum the exponentials along the same dimension
    exp_sum = torch.sum(
        exp_features,
        dim=dim,
        keepdim=True,
    )

    # 5. Divide each exponential by the sum
    return exp_features / exp_sum

#--------------------------------


def cross_entropy(
    inputs: Float[Tensor, "batch vocab_size"],
    targets: Int[Tensor, "batch"],
) -> Float[Tensor, ""]:
    """
    Compute the multi-class cross-entropy loss.

    inputs:
        Logits before softmax
        Shape (batch, vocab_size)

    targets:
        Correct class index for each example
        Shape (batch,)

    return:
        Mean loss over the batch, a scalar
    """

    # 1. Subtract each row's max to avoid exp overflow
    max_logits = inputs.max(
        dim=-1,
        keepdim=True,
    ).values

    shifted_logits = inputs - max_logits

    # 2. Gather the logit of the correct class for each example
    #
    # targets.unsqueeze(-1):
    # (batch,) -> (batch, 1)
    #
    # After gather:
    # (batch, 1) -> squeeze -> (batch,)
    target_logits = shifted_logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # 3. log(sum(exp(logits))) for each row
    log_sum_exp = torch.log(
        torch.exp(shifted_logits).sum(dim=-1)
    )

    # 4. Per-example cross-entropy
    losses = -target_logits + log_sum_exp

    # 5. Average over the batch and return a scalar
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
    Clip gradients by their global L2 norm across all parameters.

    parameters:
        Model parameters

    max_l2_norm:
        Maximum allowed global gradient L2 norm
    """

    # 1. Keep only parameters that have a gradient
    grads = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]

    # Nothing to do when there are no gradients
    if len(grads) == 0:
        return

    # 2. Global L2 norm of all gradients together
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

    # 3. Leave gradients unchanged when the total norm is within the limit
    if total_norm <= max_l2_norm:
        return

    # 4. Every gradient shares the same scale factor
    scale = max_l2_norm / (total_norm + 1e-6)

    # 5. Scale each parameter's existing grad in place
    with torch.no_grad():
        for grad in grads:
            grad.mul_(scale)