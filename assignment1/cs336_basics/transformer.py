from cs336_basics.attention import MultiHeadSelfAttention
# [original code, commented out]
# from cs336_basics.layers import RMSNorm, SwiGLU, Embedding, Linear

# [added: SiLU FFN]
from cs336_basics.layers import RMSNorm, SwiGLU, SiLUFFN, Embedding, Linear

import torch
from torch import Tensor
from torch import nn



class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.

    Data flow:

        x
        -> RMSNorm
        -> causal MHA with RoPE
        -> residual add
        -> RMSNorm
        -> SwiGLU
        -> residual add

    Input:
        (..., seq_len, d_model)

    Output:
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
        disable_rmsnorm: bool = False,  # [added] RMSNorm stays enabled by default
        use_post_norm: bool = False,  # [added: post-norm] off by default
        disable_rope: bool = False,  # [added: NoPE] off by default; True disables RoPE
        # [added: SiLU FFN] first site: accept the argument
        use_silu_ffn: bool = False,
    ):
        super().__init__()
        # [added] store the switch
        self.disable_rmsnorm = disable_rmsnorm
        # [added: post-norm] store the architecture switch
        self.use_post_norm = use_post_norm
        # [added: NoPE] store the switch
        self.disable_rope = disable_rope

        # [added: SiLU FFN] second site: store the argument
        self.use_silu_ffn = use_silu_ffn

        """

        # [added: post-norm] prevent enabling both ablations at once
        if use_post_norm and disable_rmsnorm:
            raise ValueError(
                "use_post_norm and disable_rmsnorm cannot both be enabled"
            )

        """

        # [added: NoPE] at most one of the three ablation switches may be on
        # When all are off, the original pre-norm + RMSNorm + RoPE path is used
        # [original code, commented out]
        # if sum((disable_rmsnorm, use_post_norm, disable_rope)) > 1:
        #     raise ValueError(
        #         "At most one of disable_rmsnorm, use_post_norm, and disable_rope "
        #         "may be enabled, so each run changes only one ablation"
        #     )

        # [added: SiLU FFN] at most one of the four ablation switches may be on
        if sum((disable_rmsnorm, use_post_norm, disable_rope, use_silu_ffn)) > 1:
            raise ValueError(
                "At most one of disable_rmsnorm, use_post_norm, disable_rope, "
                "and use_silu_ffn may be enabled, so each run changes only one ablation"
            )

        # RMSNorm before the attention branch
        # [added: post-norm] RMSNorm for attention; the switch decides where it runs
        self.ln1 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        # Multi-head self-attention with RoPE and a causal mask
        # [added: post-norm] RMSNorm for the FFN; the switch decides where it runs
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            # [original code, commented out: always enable RoPE]
            # theta=theta,
            # max_seq_len=max_seq_len,

            # [added: NoPE] pass None for both to use Attention's existing no-RoPE path
            # When the switch is off, still pass the original theta and max_seq_len
            theta=None if self.disable_rope else theta,
            max_seq_len=None if self.disable_rope else max_seq_len,

            device=device,
            dtype=dtype,
        )

        # ===== [fix 1: restore ln2] start =====
        # RMSNorm for the FFN.
        # forward still uses self.ln2, so it must be created here.
        # Whether it runs before the FFN or after the residual add is decided by forward.
        self.ln2 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        # ===== [fix 1: restore ln2] end =====

        # [original code, commented out: always use SwiGLU]
        # self.ffn = SwiGLU(
        #     d_model=d_model,
        #     d_ff=d_ff,
        #     device=device,
        #     dtype=dtype,
        # )

        # ===== [added: SiLU FFN] start =====
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
        # ===== [added: SiLU FFN] end =====

        # Feed-forward network
        # original code, commented out
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

        If omitted, MHA defaults internally to:
            0, 1, ..., seq_len - 1
        """

        # ===== [added: post-norm] start =====
        if self.use_post_norm:
            # Part 1: Attention -> residual add -> RMSNorm
            # residual saves x before the attention sublayer
            residual = x

            attention_output = self.attn(
                x,
                token_positions=token_positions,
            )

            x = residual + attention_output
            x = self.ln1(x)

            # Part 2: SwiGLU -> residual add -> RMSNorm
            # x here has already passed through ln1 above
            # residual saves x before the FFN sublayer
            residual = x

            ffn_output = self.ffn(x)

            x = residual + ffn_output
            x = self.ln2(x)

            # This branch has finished the whole block, so skip the pre-norm path below
            return x
        # ===== [added: post-norm] end =====

        # Part 1:
        # RMSNorm -> Attention -> residual connection
        residual = x

        # [original code, commented out]
        # normalized_x = self.ln1(x)

        # [added] skip RMSNorm when the switch is on
        if self.disable_rmsnorm:
            normalized_x = x
        else:
            normalized_x = self.ln1(x)

        attention_output = self.attn(
            normalized_x,
            token_positions=token_positions,
        )

        x = residual + attention_output

        # Part 2:
        # RMSNorm -> SwiGLU -> residual connection
        residual = x

        # [original code, commented out]
        # normalized_x = self.ln2(x)

        # [added] skip RMSNorm when the switch is on
        if self.disable_rmsnorm:
            normalized_x = x
        else:
            normalized_x = self.ln2(x)

        ffn_output = self.ffn(normalized_x)

        x = residual + ffn_output

        return x




class TransformerLM(nn.Module):
    """
    Decoder-only Transformer language model.

    Data flow:

        token IDs
        -> token embedding
        -> TransformerBlock × num_layers
        -> final RMSNorm
        -> lm_head
        -> logits

    Input:
        (batch, seq_len)

    Output:
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
        disable_rmsnorm: bool = False,  # [added] RMSNorm stays enabled by default
        use_post_norm: bool = False,  # [added: post-norm] off by default
        disable_rope: bool = False,  # [added: NoPE] off by default; True disables RoPE
        # [added: SiLU FFN] accept the switch passed in by the training script
        use_silu_ffn: bool = False,
    ):
        super().__init__()
        
        # [added] store the switch
        self.disable_rmsnorm = disable_rmsnorm

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        # 1. token id -> token embedding
        #
        # Weight:
        # (vocab_size, d_model)
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # 2. Stack of TransformerBlocks
        #
        # Parameter names become:
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
                    # [added] pass the same switch to every Block
                    disable_rmsnorm=disable_rmsnorm,
                    use_post_norm=use_post_norm,  # [added: post-norm]
                    disable_rope=disable_rope,  # [added: NoPE] pass it to every Block
                    # [added: SiLU FFN] pass it to this Block
                    use_silu_ffn=use_silu_ffn,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Final RMSNorm
        self.ln_final = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        # 4. hidden state -> vocabulary logits
        #
        # Weight:
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

        # 1. Look up embeddings
        #
        # (batch, seq_len)
        #     ->
        # (batch, seq_len, d_model)
        x = self.token_embeddings(in_indices)

        # Build positions from the actual input length, not context_length
        #
        # If:
        # context_length = 16
        # seq_len = 8
        #
        # this produces 0..7, not 0..15
        token_positions = torch.arange(
            seq_len,
            device=in_indices.device,
        )

        # 2. Pass through each TransformerBlock in order
        #
        # Each layer keeps the shape:
        # (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(
                x,
                token_positions=token_positions,
            )

        # 3. Final RMSNorm
        #
        # (batch, seq_len, d_model)
        # [original code, commented out]
        # x = self.ln_final(x)

        # [added] run the final RMSNorm only when it is not disabled
        if not self.disable_rmsnorm:
            x = self.ln_final(x)

        # 4. Project onto the full vocabulary
        #
        # (batch, seq_len, d_model)
        #     ->
        # (batch, seq_len, vocab_size)
        logits = self.lm_head(x)

        # Do not call softmax here
        return logits