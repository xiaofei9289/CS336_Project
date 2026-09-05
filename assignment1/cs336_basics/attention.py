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
    计算 Scaled Dot-Product Attention。

    Attention(Q, K, V)
        = softmax(QK^T / sqrt(d_k)) V

    参数形状：
        query: (..., queries, d_k)
        key:   (..., keys, d_k)
        value: (..., keys, d_v)
        mask:  (..., queries, keys)，或者可以广播到该形状

    返回形状：
        (..., queries, d_v)
    """

    # Q 和 K 最后一维必须相同，
    # 因为它们需要进行点积
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "query 和 key 的最后一维必须相同，"
            f"但得到了 {query.shape[-1]} 和 {key.shape[-1]}"
        )

    # Key 的数量必须与 Value 的数量相同
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(
            "key 和 value 的序列长度必须相同，"
            f"但得到了 {key.shape[-2]} 和 {value.shape[-2]}"
        )

    # 1. 取得 d_k
    d_k = query.shape[-1]

    # 2. 计算每个 Query 和每个 Key 的点积分数
    #
    # query:             (..., queries, d_k)
    # key.transpose:     (..., d_k, keys)
    # attention_scores:  (..., queries, keys)
    attention_scores = (
        query @ key.transpose(-2, -1)
    )

    # 3. 除以 sqrt(d_k)，防止分数过大
    attention_scores = attention_scores / math.sqrt(d_k)

    # 4. 在 Softmax 之前应用 mask
    if mask is not None:
        mask = mask.to(
            device=attention_scores.device,
            dtype=torch.bool,
        )

        # True  = 可以看
        # False = 不可以看
        attention_scores = attention_scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    # 5. 对 keys 这一维做 Softmax
    #
    # attention_weights:
    # (..., queries, keys)
    attention_weights = softmax(
        attention_scores,
        dim=-1,
    )

    # 6. 根据注意力权重对 Value 加权求和
    #
    # attention_weights: (..., queries, keys)
    # value:              (..., keys, d_v)
    # output:             (..., queries, d_v)
    output = attention_weights @ value

    return output



# class MultiHeadSelfAttention(nn.Module):
#     """
#     没有 RoPE、没有 causal mask 的多头自注意力。

#     输入形状：
#         (..., seq_len, d_model)

#     输出形状：
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

#         # d_model 必须能够平均分给每一个 head
#         if d_model % num_heads != 0:
#             raise ValueError(
#                 f"d_model={d_model} must be divisible by "
#                 f"num_heads={num_heads}"
#             )

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads

#         # Q、K、V 一次性投影出所有 head
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

#         # 拼接所有 head 后的输出投影
#         self.o_proj = Linear(
#             in_features=d_model,
#             out_features=d_model,
#             device=device,
#             dtype=dtype,
#         )

#     def split_heads(self, x: Tensor) -> Tensor:
#         """
#         把最后一个 d_model 维拆成：

#             num_heads × head_dim

#         输入：
#             (..., seq_len, d_model)

#         输出：
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

#         # 交换 seq_len 和 num_heads
#         #
#         # (..., seq_len, num_heads, head_dim)
#         #             ↓
#         # (..., num_heads, seq_len, head_dim)
#         x = x.transpose(-3, -2)

#         return x

#     def merge_heads(self, x: Tensor) -> Tensor:
#         """
#         把多个 head 重新拼接成 d_model。

#         输入：
#             (..., num_heads, seq_len, head_dim)

#         输出：
#             (..., seq_len, d_model)
#         """

#         leading_shape = x.shape[:-3]
#         seq_len = x.shape[-2]

#         # split_heads 中 transpose 的逆操作
#         #
#         # (..., num_heads, seq_len, head_dim)
#         #             ↓
#         # (..., seq_len, num_heads, head_dim)
#         x = x.transpose(-3, -2)

#         # transpose 后张量可能不连续
#         x = x.contiguous()

#         # 拼接 num_heads 和 head_dim
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
#         输入：
#             x: (..., seq_len, d_model)

#         输出：
#             (..., seq_len, d_model)
#         """

#         seq_len = x.shape[-2]

#         # 1. 一次性生成全部 head 的 Q、K、V
#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)

#         # 2. 拆分 head
#         #
#         # (..., seq_len, d_model)
#         #     ->
#         # (..., num_heads, seq_len, head_dim)
#         q = self.split_heads(q)
#         k = self.split_heads(k)
#         v = self.split_heads(v)

#         # 3. 创建 causal mask
#         #
#         # True：允许关注
#         # False：禁止关注
#         #
#         # seq_len = 3 时：
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

#         # causal_mask 的形状是：
#         #
#         # (seq_len, seq_len)
#         #
#         # 它会自动广播到：
#         #
#         # (..., num_heads, seq_len, seq_len)
#         attention_output = scaled_dot_product_attention(
#             q,
#             k,
#             v,
#             mask=causal_mask,
#         )

#         # 4. 拼回所有 head
#         attention_output = self.merge_heads(attention_output)

#         # 5. 输出投影
#         output = self.o_proj(attention_output)

#         return output



class MultiHeadSelfAttention(nn.Module):
    """
    同时支持：

    1. 普通 causal MHA
    2. 带 RoPE 的 causal MHA

    输入：
        (..., seq_len, d_model)

    输出：
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

        # 一次性生成所有 head 的 Q、K、V
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

        # theta 和 max_seq_len 都传入时，启用 RoPE
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,

                # 注意：这里是每个 head 的维度，
                # 不是整个 d_model
                d_k=self.head_dim,

                max_seq_len=max_seq_len,
                device=device,
            )
        elif theta is None and max_seq_len is None:
            # 普通、不带 RoPE 的 MHA
            self.rope = None
        else:
            raise ValueError(
                "theta and max_seq_len must either both be provided "
                "or both be None"
            )

    def split_heads(self, x: Tensor) -> Tensor:
        """
        输入：
            (..., seq_len, d_model)

        输出：
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
        输入：
            (..., num_heads, seq_len, head_dim)

        输出：
            (..., seq_len, d_model)
        """

        leading_shape = x.shape[:-3]
        seq_len = x.shape[-2]

        # (..., num_heads, seq_len, head_dim)
        #     ->
        # (..., seq_len, num_heads, head_dim)
        x = x.transpose(-3, -2)

        # transpose 后通常不是连续内存
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

        例如测试可能传入：
            (1, seq_len)

        不应假设 token_positions 一定是：
            [0, 1, 2, ..., seq_len - 1]
        """

        seq_len = x.shape[-2]

        # 1. Q、K、V 一次性投影
        #
        # (..., seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 拆分 head
        #
        # (..., num_heads, seq_len, head_dim)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 3. 仅对 Q、K 使用 RoPE
        #
        # V 不进行旋转
        if self.rope is not None:
            if token_positions is None:
                # 只有未显式传入位置时，才默认使用连续位置
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

        # 4. 创建 causal mask
        #
        # True：允许关注
        # False：不能关注
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )
        )

        # 5. 每个 head 计算 SDPA
        #
        # (..., num_heads, seq_len, head_dim)
        attention_output = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=causal_mask,
        )

        # 6. 拼接所有 head
        #
        # (..., seq_len, d_model)
        attention_output = self.merge_heads(
            attention_output
        )

        # 7. 输出投影
        output = self.output_proj(attention_output)

        return output