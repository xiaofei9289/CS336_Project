import torch
from torch import Tensor, nn
from jaxtyping import Float, Int


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(
                f"d_k must be even, but got d_k={d_k}"
            )

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Corresponding dimensions:
        # 0, 2, 4, ..., d_k - 2
        #
        # Each pair of adjacent dimensions shares one rotation frequency:
        # (x0, x1), (x2, x3), ...
        pair_indices = torch.arange(
            0,
            d_k,
            2,
            device=device,
            dtype=torch.float32,
        )

        # ω_i = theta^(-2i / d_k)
        #
        # pair_indices is already:
        # 0, 2, 4, ...
        # so divide directly by d_k
        inverse_frequencies = theta ** (
            -pair_indices / d_k
        )

        # Positions:
        # 0, 1, 2, ..., max_seq_len - 1
        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        # Rotation angle for each position and dimension pair
        #
        # positions:           (max_seq_len,)
        # inverse_frequencies: (d_k / 2,)
        #
        # angles:              (max_seq_len, d_k / 2)
        angles = positions[:, None] * inverse_frequencies[None, :]

        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)

        # cos/sin are not trainable, but they should move with the model device
        self.register_buffer(
            "cos_cached",
            cos_cached,
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            sin_cached,
            persistent=False,
        )

    def forward(
        self,
        in_query_or_key: Float[Tensor, "... sequence_length d_k"],
        token_positions: Int[Tensor, "... sequence_length"],
    ) -> Float[Tensor, "... sequence_length d_k"]:

        if in_query_or_key.shape[-1] != self.d_k:
            raise ValueError(
                "The last input dimension must equal d_k, "
                f"but the input is {in_query_or_key.shape[-1]}, "
                f"d_k={self.d_k}"
            )

        # Keep position indices on the same device as the cache
        token_positions = token_positions.to(
            device=self.cos_cached.device
        )

        # Look up cos/sin for the actual token_positions
        #
        # For example, token_positions = [3, 7, 2]
        # selects cache rows 3, 7, and 2
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # Keep the output dtype equal to the input dtype, for example BF16
        cos = cos.to(
            device=in_query_or_key.device,
            dtype=in_query_or_key.dtype,
        )

        sin = sin.to(
            device=in_query_or_key.device,
            dtype=in_query_or_key.dtype,
        )

        # If the input has a head dimension and token_positions does not,
        # insert a singleton dimension before sequence_length so they broadcast.
        #
        # For example:
        # input: (batch, heads, seq, d_k)
        # cos:   (batch, seq, d_k/2)
        #
        # becomes: (batch, 1, seq, d_k/2)
        while cos.ndim < in_query_or_key.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # Split each 2D pair
        #
        # even: x0, x2, x4, ...
        # odd:  x1, x3, x5, ...
        even = in_query_or_key[..., 0::2]
        odd = in_query_or_key[..., 1::2]

        # 2D rotation:
        #
        # x_even' = x_even cos - x_odd sin
        # x_odd'  = x_even sin + x_odd cos
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        # First stack into:
        # (..., sequence_length, d_k / 2, 2)
        #
        # then flatten back to:
        # (..., sequence_length, d_k)
        rotated = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        )

        return rotated.flatten(start_dim=-2)