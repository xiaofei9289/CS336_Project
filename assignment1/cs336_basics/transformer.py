from cs336_basics.attention import MultiHeadSelfAttention
# [原代码，已注释]
# from cs336_basics.layers import RMSNorm, SwiGLU, Embedding, Linear

# [新增：SiLU FFN]
from cs336_basics.layers import RMSNorm, SwiGLU, SiLUFFN, Embedding, Linear

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
        disable_rmsnorm: bool = False,  # [新增] 默认不禁用 RMSNorm
        use_post_norm: bool = False,  # [新增：post-norm] 默认关闭
        disable_rope: bool = False,  # [新增：NoPE] 默认关闭；True 时禁用 RoPE
        # [新增：SiLU FFN] 第一处：接收参数
        use_silu_ffn: bool = False,
    ):
        super().__init__()
        # [新增] 保存开关
        self.disable_rmsnorm = disable_rmsnorm
        # [新增：post-norm] 保存架构开关
        self.use_post_norm = use_post_norm
        # [新增：NoPE] 保存开关
        self.disable_rope = disable_rope

        # [新增：SiLU FFN] 第二处：保存参数
        self.use_silu_ffn = use_silu_ffn

        """

        # [新增：post-norm] 防止同时启用两项消融
        if use_post_norm and disable_rmsnorm:
            raise ValueError(
                "use_post_norm 和 disable_rmsnorm 不能同时开启"
            )

        """

        # [新增：NoPE] 三个消融开关最多只能开启一个
        # 全部关闭时，仍使用原来的 pre-norm + RMSNorm + RoPE
        # [原代码，已注释]
        # if sum((disable_rmsnorm, use_post_norm, disable_rope)) > 1:
        #     raise ValueError(
        #         "disable_rmsnorm、use_post_norm、disable_rope "
        #         "最多只能开启一个，以保证每次只做一项消融"
        #     )

        # [新增：SiLU FFN] 四个消融开关最多开启一个
        if sum((disable_rmsnorm, use_post_norm, disable_rope, use_silu_ffn)) > 1:
            raise ValueError(
                "disable_rmsnorm、use_post_norm、disable_rope、use_silu_ffn "
                "最多只能开启一个，以保证每次只做一项消融"
            )

        # Attention 分支前的 RMSNorm
        # [新增：post-norm] Attention 对应的 RMSNorm，执行位置由开关决定
        self.ln1 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        # 带 RoPE、causal mask 的多头自注意力
        # [新增：post-norm] FFN 对应的 RMSNorm，执行位置由开关决定
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            # [原代码，已注释：始终启用 RoPE]
            # theta=theta,
            # max_seq_len=max_seq_len,

            # [新增：NoPE] 同时传 None，使用 Attention 已有的无 RoPE 路径
            # 开关关闭时，仍传入原来的 theta 和 max_seq_len
            theta=None if self.disable_rope else theta,
            max_seq_len=None if self.disable_rope else max_seq_len,

            device=device,
            dtype=dtype,
        )

        # ===== [修复 1：补回 ln2] 开始 =====
        # FFN 对应的 RMSNorm。
        # forward 中仍然使用 self.ln2，因此这里必须创建。
        # 具体在 FFN 前还是残差相加后执行，由原来的 forward 决定。
        self.ln2 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        # ===== [修复 1：补回 ln2] 结束 =====

        # [原代码，已注释：始终使用 SwiGLU]
        # self.ffn = SwiGLU(
        #     d_model=d_model,
        #     d_ff=d_ff,
        #     device=device,
        #     dtype=dtype,
        # )

        # ===== [新增：SiLU FFN] 开始 =====
        if self.use_silu_ffn:
            self.ffn = SiLUFFN(
                d_model=d_model,
                d_ff=d_ff,
                device=device,
                dtype=dtype,
            )
        else:
            self.ffn = SwiGLU(
                d_model=d_model,
                d_ff=d_ff,
                device=device,
                dtype=dtype,
            )
        # ===== [新增：SiLU FFN] 结束 =====

        # Feed-forward network
        # 原代码注释
        # self.ffn = SwiGLU(
        #     d_model=d_model,
        #     d_ff=d_ff,
        #     device=device,
        #     dtype=dtype,
        # )

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

        # ===== [新增：post-norm] 开始 =====
        if self.use_post_norm:
            # 第一部分：Attention -> 加残差 -> RMSNorm
            # residual 保存进入 Attention 子层之前的 x
            residual = x

            attention_output = self.attn(
                x,
                token_positions=token_positions,
            )

            x = residual + attention_output
            x = self.ln1(x)

            # 第二部分：SwiGLU -> 加残差 -> RMSNorm
            # 此时的 x 已经过上面的 ln1
            # residual 保存进入 FFN 子层之前的 x
            residual = x

            ffn_output = self.ffn(x)

            x = residual + ffn_output
            x = self.ln2(x)

            # 本分支已经完成整个 Block，不再执行下面的 pre-norm
            return x
        # ===== [新增：post-norm] 结束 =====

        # 第一部分：
        # RMSNorm -> Attention -> 残差连接
        residual = x

        # [原代码，已注释]
        # normalized_x = self.ln1(x)

        # [新增] 根据开关决定是否跳过 RMSNorm
        if self.disable_rmsnorm:
            normalized_x = x
        else:
            normalized_x = self.ln1(x)

        attention_output = self.attn(
            normalized_x,
            token_positions=token_positions,
        )

        x = residual + attention_output

        # 第二部分：
        # RMSNorm -> SwiGLU -> 残差连接
        residual = x

        # [原代码，已注释]
        # normalized_x = self.ln2(x)

        # [新增] 根据开关决定是否跳过 RMSNorm
        if self.disable_rmsnorm:
            normalized_x = x
        else:
            normalized_x = self.ln2(x)

        ffn_output = self.ffn(normalized_x)

        x = residual + ffn_output

        return x




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
        disable_rmsnorm: bool = False,  # [新增] 默认不禁用 RMSNorm
        use_post_norm: bool = False,  # [新增：post-norm] 默认关闭
        disable_rope: bool = False,  # [新增：NoPE] 默认关闭；True 时禁用 RoPE
        # [新增：SiLU FFN] 接收训练脚本传入的开关
        use_silu_ffn: bool = False,
    ):
        super().__init__()
        
        # [新增] 保存开关
        self.disable_rmsnorm = disable_rmsnorm

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
                    # [新增] 把同一个开关传给每一层 Block
                    disable_rmsnorm=disable_rmsnorm,
                    use_post_norm=use_post_norm,  # [新增：post-norm]
                    disable_rope=disable_rope,  # [新增：NoPE] 传给每一层 Block
                    # [新增：SiLU FFN] 传给当前这一层 Block
                    use_silu_ffn=use_silu_ffn,
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
        # [原代码，已注释]
        # x = self.ln_final(x)

        # [新增] 未禁用时才执行最终 RMSNorm
        if not self.disable_rmsnorm:
            x = self.ln_final(x)

        # 4. 映射到整个词表
        #
        # (batch, seq_len, d_model)
        #     ->
        # (batch, seq_len, vocab_size)
        logits = self.lm_head(x)

        # 不要在这里调用 softmax
        return logits