from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.layers import RMSNorm, SwiGLU, Embedding, Linear

import torch
from torch import Tensor
from torch import nn


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block。

    数据流：

        x
        -> RMSNorm
        -> causal MHA with RoPE
        -> residual add
        -> RMSNorm
        -> SwiGLU
        -> residual add

    输入：
        (..., seq_len, d_model)

    输出：
        (..., seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # Attention 分支前的 RMSNorm
        self.ln1 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        # 带 RoPE、causal mask 的多头自注意力
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        # FFN 分支前的 RMSNorm
        self.ln2 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        # Feed-forward network
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

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

        如果没有传入，MHA 内部默认使用：
            0, 1, ..., seq_len - 1
        """

        # 第一部分：
        # RMSNorm -> Attention -> 残差连接
        residual = x

        normalized_x = self.ln1(x)

        attention_output = self.attn(
            normalized_x,
            token_positions=token_positions,
        )

        x = residual + attention_output

        # 第二部分：
        # RMSNorm -> SwiGLU -> 残差连接
        residual = x

        normalized_x = self.ln2(x)

        ffn_output = self.ffn(normalized_x)

        x = residual + ffn_output

        return x

import torch
from torch import Tensor
from torch import nn


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer language model。

    数据流：

        token IDs
        -> token embedding
        -> TransformerBlock × num_layers
        -> final RMSNorm
        -> lm_head
        -> logits

    输入：
        (batch, seq_len)

    输出：
        (batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        # 1. token id -> token embedding
        #
        # 权重：
        # (vocab_size, d_model)
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # 2. 多层 TransformerBlock
        #
        # 参数名称自动变成：
        # layers.0.xxx
        # layers.1.xxx
        # ...
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. 最后的 RMSNorm
        self.ln_final = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        # 4. hidden state -> vocabulary logits
        #
        # 权重：
        # (vocab_size, d_model)
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        in_indices: Tensor,
    ) -> Tensor:
        """
        in_indices:
            (batch, seq_len)

        return:
            (batch, seq_len, vocab_size)
        """

        seq_len = in_indices.shape[-1]

        if seq_len > self.context_length:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds "
                f"context length {self.context_length}"
            )

        # 1. 查 embedding
        #
        # (batch, seq_len)
        #     ->
        # (batch, seq_len, d_model)
        x = self.token_embeddings(in_indices)

        # 使用实际输入长度生成位置，而不是 context_length
        #
        # 如果：
        # context_length = 16
        # seq_len = 8
        #
        # 那么这里生成 0..7，而不是 0..15
        token_positions = torch.arange(
            seq_len,
            device=in_indices.device,
        )

        # 2. 依次经过每一个 TransformerBlock
        #
        # 每层形状保持：
        # (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(
                x,
                token_positions=token_positions,
            )

        # 3. 最后的 RMSNorm
        #
        # (batch, seq_len, d_model)
        x = self.ln_final(x)

        # 4. 映射到整个词表
        #
        # (batch, seq_len, d_model)
        #     ->
        # (batch, seq_len, vocab_size)
        logits = self.lm_head(x)

        # 不要在这里调用 softmax
        return logits