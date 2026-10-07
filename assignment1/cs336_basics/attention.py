import math

import torch
from torch import Tensor
from jaxtyping import Bool, Float
from cs336_basics.layers import softmax
from cs336_basics.layers import Linear
from torch import nn
from cs336_basics.rope import RotaryPositionalEmbedding



def scaled_dot_product_attention(
    query: Float[Tensor, "... queries d_k"],
    key: Float[Tensor, "... keys d_k"],
    value: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V)
        = softmax(QK^T / sqrt(d_k)) V

    Argument shapes:
        query: (..., queries, d_k)
        key:   (..., keys, d_k)
        value: (..., keys, d_v)
        mask:  (..., queries, keys), or broadcastable to that shape

    Return shape:
        (..., queries, d_v)
    """

    # The last dimension of Q and K must match,
    # because they are dotted together
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "The last dimensions of query and key must match, "
            f"but got {query.shape[-1]} and {key.shape[-1]}"
        )

    # The number of keys must match the number of values
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(
            "The sequence lengths of key and value must match, "
            f"but got {key.shape[-2]} and {value.shape[-2]}"
        )

    # 1. Read d_k
    d_k = query.shape[-1]

    # 2. Dot-product score between every query and every key
    #
    # query:             (..., queries, d_k)
    # key.transpose:     (..., d_k, keys)
    # attention_scores:  (..., queries, keys)
    attention_scores = (
        query @ key.transpose(-2, -1)
    )

    # 3. Divide by sqrt(d_k) so the scores do not get too large
    attention_scores = attention_scores / math.sqrt(d_k)

    # 4. Apply the mask before softmax
    if mask is not None:
        mask = mask.to(
            device=attention_scores.device,
            dtype=torch.bool,
        )

        # True  = allowed to attend
        # False = not allowed to attend
        attention_scores = attention_scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    # 5. Softmax over the keys dimension
    #
    # attention_weights:
    # (..., queries, keys)
    attention_weights = softmax(
        attention_scores,
        dim=-1,
    )

    # 6. Weighted sum of values using the attention weights
    #
    # attention_weights: (..., queries, keys)
    # value:              (..., keys, d_v)
    # output:             (..., queries, d_v)
    output = attention_weights @ value

    return output



# class MultiHeadSelfAttention(nn.Module):
#     """
#     Multi-head self-attention without RoPE and without a causal mask.

#     Input shape:
#         (..., seq_len, d_model)

#     Output shape:
#         (..., seq_len, d_model)
#     """

#     def __init__(
#         self,
#         d_model: int,
#         num_heads: int,
#         device=None,
#         dtype=None,
#     ):
#         super().__init__()

#         # d_model must be evenly divisible across heads
#         if d_model % num_heads != 0:
#             raise ValueError(
#                 f"d_model={d_model} must be divisible by "
#                 f"num_heads={num_heads}"
#             )

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads

#         # Project Q, K, and V for every head in one shot
#         self.q_proj = Linear(
#             in_features=d_model,
#             out_features=d_model,
#             device=device,
#             dtype=dtype,
#         )

#         self.k_proj = Linear(
#             in_features=d_model,
#             out_features=d_model,
#             device=device,
#             dtype=dtype,
#         )

#         self.v_proj = Linear(
#             in_features=d_model,
#             out_features=d_model,
#             device=device,
#             dtype=dtype,
#         )

#         # Output projection after concatenating every head
#         self.o_proj = Linear(
#             in_features=d_model,
#             out_features=d_model,
#             device=device,
#             dtype=dtype,
#         )

#     def split_heads(self, x: Tensor) -> Tensor:
#         """
#         Split the last d_model dimension into:

#             num_heads × head_dim

#         Input:
#             (..., seq_len, d_model)

#         Output:
#             (..., num_heads, seq_len, head_dim)
#         """

#         leading_shape = x.shape[:-2]
#         seq_len = x.shape[-2]

#         # (..., seq_len, d_model)
#         #             ↓
#         # (..., seq_len, num_heads, head_dim)
#         x = x.reshape(
#             *leading_shape,
#             seq_len,
#             self.num_heads,
#             self.head_dim,
#         )

#         # Swap seq_len and num_heads
#         #
#         # (..., seq_len, num_heads, head_dim)
#         #             ↓
#         # (..., num_heads, seq_len, head_dim)
#         x = x.transpose(-3, -2)

#         return x

#     def merge_heads(self, x: Tensor) -> Tensor:
#         """
#         Concatenate the heads back into d_model.

#         Input:
#             (..., num_heads, seq_len, head_dim)

#         Output:
#             (..., seq_len, d_model)
#         """

#         leading_shape = x.shape[:-3]
#         seq_len = x.shape[-2]

#         # Inverse of the transpose in split_heads
#         #
#         # (..., num_heads, seq_len, head_dim)
#         #             ↓
#         # (..., seq_len, num_heads, head_dim)
#         x = x.transpose(-3, -2)

#         # The tensor may be non-contiguous after transpose
#         x = x.contiguous()

#         # Concatenate num_heads and head_dim
#         #
#         # (..., seq_len, num_heads, head_dim)
#         #             ↓
#         # (..., seq_len, d_model)
#         x = x.reshape(
#             *leading_shape,
#             seq_len,
#             self.d_model,
#         )

#         return x

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Input:
#             x: (..., seq_len, d_model)

#         Output:
#             (..., seq_len, d_model)
#         """

#         seq_len = x.shape[-2]

#         # 1. Produce Q, K, and V for every head in one shot
#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)

#         # 2. Split into heads
#         #
#         # (..., seq_len, d_model)
#         #     ->
#         # (..., num_heads, seq_len, head_dim)
#         q = self.split_heads(q)
#         k = self.split_heads(k)
#         v = self.split_heads(v)

#         # 3. Build the causal mask
#         #
#         # True: attending is allowed
#         # False: attending is forbidden
#         #
#         # When seq_len = 3:
#         #
#         # [[ True, False, False],
#         #  [ True,  True, False],
#         #  [ True,  True,  True]]
#         causal_mask = torch.tril(
#             torch.ones(
#                 seq_len,
#                 seq_len,
#                 device=x.device,
#                 dtype=torch.bool,
#             )
#         )

#         # causal_mask has shape:
#         #
#         # (seq_len, seq_len)
#         #
#         # It broadcasts automatically to:
#         #
#         # (..., num_heads, seq_len, seq_len)
#         attention_output = scaled_dot_product_attention(
#             q,
#             k,
#             v,
#             mask=causal_mask,
#         )

#         # 4. Merge the heads back together
#         attention_output = self.merge_heads(attention_output)

#         # 5. Output projection
#         output = self.o_proj(attention_output)

#         return output



class MultiHeadSelfAttention(nn.Module):
    """
    Supports both:

    1. Plain causal MHA
    2. Causal MHA with RoPE

    Input:
        (..., seq_len, d_model)

    Output:
        (..., seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by "
                f"num_heads={num_heads}"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Project Q, K, and V for every head in one shot
        self.q_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.k_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.v_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.output_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        # Enable RoPE when both theta and max_seq_len are provided
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,

                # This is the per-head dimension,
                # not the full d_model
                d_k=self.head_dim,

                max_seq_len=max_seq_len,
                device=device,
            )
        elif theta is None and max_seq_len is None:
            # Plain MHA without RoPE
            self.rope = None
        else:
            raise ValueError(
                "theta and max_seq_len must either both be provided "
                "or both be None"
            )

    def split_heads(self, x: Tensor) -> Tensor:
        """
        Input:
            (..., seq_len, d_model)

        Output:
            (..., num_heads, seq_len, head_dim)
        """

        leading_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        # (..., seq_len, d_model)
        #     ->
        # (..., seq_len, num_heads, head_dim)
        x = x.reshape(
            *leading_shape,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        # (..., seq_len, num_heads, head_dim)
        #     ->
        # (..., num_heads, seq_len, head_dim)
        x = x.transpose(-3, -2)

        return x

    def merge_heads(self, x: Tensor) -> Tensor:
        """
        Input:
            (..., num_heads, seq_len, head_dim)

        Output:
            (..., seq_len, d_model)
        """

        leading_shape = x.shape[:-3]
        seq_len = x.shape[-2]

        # (..., num_heads, seq_len, head_dim)
        #     ->
        # (..., seq_len, num_heads, head_dim)
        x = x.transpose(-3, -2)

        # Memory is usually non-contiguous after transpose
        x = x.contiguous()

        # (..., seq_len, num_heads, head_dim)
        #     ->
        # (..., seq_len, d_model)
        x = x.reshape(
            *leading_shape,
            seq_len,
            self.d_model,
        )

        return x

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        """
        x:
            (..., seq_len, d_model)

        token_positions:
            (..., seq_len)

        Tests may pass, for example:
            (1, seq_len)

        Do not assume token_positions is always:
            [0, 1, 2, ..., seq_len - 1]
        """

        seq_len = x.shape[-2]

        # 1. Project Q, K, and V in one shot
        #
        # (..., seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Split into heads
        #
        # (..., num_heads, seq_len, head_dim)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 3. Apply RoPE only to Q and K
        #
        # V is not rotated
        if self.rope is not None:
            if token_positions is None:
                # Use contiguous positions only when none were passed in
                token_positions = torch.arange(
                    seq_len,
                    device=x.device,
                )

            q = self.rope(
                q,
                token_positions,
            )

            k = self.rope(
                k,
                token_positions,
            )

        # 4. Build the causal mask
        #
        # True: attending is allowed
        # False: attending is not allowed
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )
        )

        # 5. Run SDPA in every head
        #
        # (..., num_heads, seq_len, head_dim)
        attention_output = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=causal_mask,
        )

        # 6. Concatenate every head
        #
        # (..., seq_len, d_model)
        attention_output = self.merge_heads(
            attention_output
        )

        # 7. Output projection
        output = self.output_proj(attention_output)

        return output