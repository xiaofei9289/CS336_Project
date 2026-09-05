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
                f"d_k 必须是偶数，但得到了 d_k={d_k}"
            )

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 对应维度：
        # 0, 2, 4, ..., d_k - 2
        #
        # 每两个相邻维度共享一个旋转频率：
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
        # 因为 pair_indices 已经是：
        # 0, 2, 4, ...
        # 所以直接除以 d_k
        inverse_frequencies = theta ** (
            -pair_indices / d_k
        )

        # 位置：
        # 0, 1, 2, ..., max_seq_len - 1
        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        # 每个位置、每个维度对的旋转角度
        #
        # positions:           (max_seq_len,)
        # inverse_frequencies: (d_k / 2,)
        #
        # angles:              (max_seq_len, d_k / 2)
        angles = positions[:, None] * inverse_frequencies[None, :]

        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)

        # cos/sin 不是可训练参数，但需要跟随模型移动设备
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
                "输入最后一维必须等于 d_k，"
                f"但输入为 {in_query_or_key.shape[-1]}，"
                f"d_k={self.d_k}"
            )

        # 确保位置索引和缓存位于同一设备
        token_positions = token_positions.to(
            device=self.cos_cached.device
        )

        # 根据真实 token_positions 取对应的 cos/sin
        #
        # 例如 token_positions = [3, 7, 2]
        # 就分别取缓存中的第 3、7、2 行
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 保持输出 dtype 与输入一致，例如 BF16
        cos = cos.to(
            device=in_query_or_key.device,
            dtype=in_query_or_key.dtype,
        )

        sin = sin.to(
            device=in_query_or_key.device,
            dtype=in_query_or_key.dtype,
        )

        # 如果输入中存在 head 维，而 token_positions 没有，
        # 在 sequence_length 前补单例维度以支持广播。
        #
        # 例如：
        # 输入：(batch, heads, seq, d_k)
        # cos：(batch, seq, d_k/2)
        #
        # 补成：(batch, 1, seq, d_k/2)
        while cos.ndim < in_query_or_key.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # 拆出每个二维 pair
        #
        # even: x0, x2, x4, ...
        # odd:  x1, x3, x5, ...
        even = in_query_or_key[..., 0::2]
        odd = in_query_or_key[..., 1::2]

        # 二维旋转：
        #
        # x_even' = x_even cos - x_odd sin
        # x_odd'  = x_even sin + x_odd cos
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        # 先组合成：
        # (..., sequence_length, d_k / 2, 2)
        #
        # 然后还原为：
        # (..., sequence_length, d_k)
        rotated = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        )

        return rotated.flatten(start_dim=-2)