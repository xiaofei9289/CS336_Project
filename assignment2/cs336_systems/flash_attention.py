"""CS336 Assignment 2: FlashAttention forward (a)-(c) and flash_backward.

Public classes returned by tests/adapters.py:
    FlashAttentionPyTorch: tiled PyTorch forward, compiled PyTorch backward.
    FlashAttentionTriton: tiled Triton forward, compiled PyTorch backward.

Both accept .apply(Q, K, V, is_causal=False), return O, and save L/Q/K/V/O.
Inputs have shape (..., sequence_length, d); Q and K may have different lengths.
Causal attention permits key j iff query i >= j (top-left alignment).
Following the handout, masked scores receive an additive -1e6 penalty.

The backward follows the handout's dense recomputation equations. It may
materialize quadratic-size intermediate tensors; it is NOT a Triton backward.
Warm up forward AND backward before timing to exclude torch.compile startup.
"""

import math

import torch
import triton
import triton.language as tl


def validate_inputs(Q, K, V):
    if min(Q.ndim, K.ndim, V.ndim) < 2:
        raise ValueError("Q、K、V 至少需要两个维度")
    if Q.shape[:-2] != K.shape[:-2] or K.shape[:-2] != V.shape[:-2]:
        raise ValueError("Q、K、V 的 batch 维度必须一致")
    if Q.shape[-1] != K.shape[-1] or K.shape[-1] != V.shape[-1]:
        raise ValueError("本实现要求 Q、K、V 的最后一维相同")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError("K 和 V 的序列长度必须一致")
    if Q.device != K.device or Q.device != V.device:
        raise ValueError("Q、K、V 必须位于同一设备")
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError("Q、K、V 必须使用相同 dtype")
    if not Q.is_floating_point():
        raise ValueError("Q、K、V 必须是浮点张量")
    if Q.numel() == 0 or K.numel() == 0 or V.numel() == 0:
        raise ValueError("输入不能为空")


# ============================================================
# flash_backward：PyTorch 公式 + torch.compile，不写 Triton backward
# ============================================================

@torch.compile
def flash_backward_pytorch(Q, K, V, O, dO, L, is_causal):
    # FP16/BF16/FP32 输入的中间计算采用 FP32。
    # 纯 PyTorch 前向也支持 FP64，因此这里保留 FP64。
    compute_dtype = (
        torch.float64 if Q.dtype == torch.float64 else torch.float32
    )
    q = Q.to(compute_dtype)
    k = K.to(compute_dtype)
    v = V.to(compute_dtype)
    o = O.to(compute_dtype)
    do = dO.to(compute_dtype)
    logsumexp = L.to(compute_dtype)
    scale = 1.0 / math.sqrt(Q.shape[-1])

    # D_i = rowsum(O * dO)，每个 query 行对应一个值。
    D = (o * do).sum(dim=-1)

    # 重计算分数 S，以及前向未保存的注意力概率 P。
    S = (q @ k.transpose(-2, -1)) * scale
    if is_causal:
        query_indices = torch.arange(Q.shape[-2], device=Q.device)
        key_indices = torch.arange(K.shape[-2], device=Q.device)
        allowed = query_indices[:, None] >= key_indices[None, :]
        S = S + torch.where(allowed, 0.0, -1e6)
    P = torch.exp(S - logsumexp.unsqueeze(-1))

    dV = P.transpose(-2, -1) @ do
    dP = do @ v.transpose(-2, -1)
    dS = P * (dP - D.unsqueeze(-1))
    dQ = (dS @ k) * scale
    dK = (dS.transpose(-2, -1) @ q) * scale

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


# ============================================================
# flash_forward (a)：纯 PyTorch 分块前向
# ============================================================

class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        validate_inputs(Q, K, V)
        batch_shape = Q.shape[:-2]
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        d = Q.shape[-1]
        query_tile_size = 16
        key_tile_size = 16
        scale = 1.0 / math.sqrt(d)
        acc_dtype = (
            torch.float64 if Q.dtype == torch.float64 else torch.float32
        )

        O = torch.empty_like(Q)
        L = torch.empty(
            (*batch_shape, n_queries), device=Q.device, dtype=acc_dtype
        )

        for q_start in range(0, n_queries, query_tile_size):
            q_end = min(q_start + query_tile_size, n_queries)
            rows = q_end - q_start
            q_block = Q[..., q_start:q_end, :].to(acc_dtype)

            m = torch.full(
                (*batch_shape, rows), float("-inf"),
                device=Q.device, dtype=acc_dtype,
            )
            l = torch.zeros_like(m)
            acc = torch.zeros(
                (*batch_shape, rows, d), device=Q.device, dtype=acc_dtype
            )

            for k_start in range(0, n_keys, key_tile_size):
                k_end = min(k_start + key_tile_size, n_keys)
                k_block = K[..., k_start:k_end, :].to(acc_dtype)
                v_block = V[..., k_start:k_end, :].to(acc_dtype)
                scores = (q_block @ k_block.transpose(-2, -1)) * scale

                # 前向、反向使用相同的因果规则。
                if is_causal:
                    query_indices = torch.arange(q_start, q_end, device=Q.device)
                    key_indices = torch.arange(k_start, k_end, device=Q.device)
                    allowed = query_indices[:, None] >= key_indices[None, :]
                    scores = scores + torch.where(allowed, 0.0, -1e6)

                block_max = scores.max(dim=-1).values
                m_new = torch.maximum(m, block_max)
                alpha = torch.exp(m - m_new)
                p = torch.exp(scores - m_new.unsqueeze(-1))
                l = alpha * l + p.sum(dim=-1)
                acc = alpha.unsqueeze(-1) * acc + p @ v_block
                m = m_new

            O[..., q_start:q_end, :] = (acc / l.unsqueeze(-1)).to(Q.dtype)
            L[..., q_start:q_end] = m + torch.log(l)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_backward_pytorch(
            Q, K, V, O, grad_output, L, ctx.is_causal
        )
        # Bool 参数没有梯度；兼容 apply(Q,K,V) 和 apply(Q,K,V,True)。
        gradients = (dQ, dK, dV, None)
        return gradients[:len(ctx.needs_input_grad)]


# ============================================================
# flash_forward (b)(c)：Triton 分块前向 + 因果 mask
# ============================================================

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    D_BLOCK: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    q_start = query_tile_index * Q_TILE_SIZE

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D_BLOCK),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_BLOCK),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_BLOCK),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D_BLOCK),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    query_indices = q_start + tl.arange(0, Q_TILE_SIZE)
    key_offsets = tl.arange(0, K_TILE_SIZE)
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D_BLOCK), dtype=tl.float32)

    # 一个程序实例固定一块 Q，只循环遍历 K/V 块。
    for tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        key_indices = tile_index * K_TILE_SIZE + key_offsets

        # -inf 仅用于排除越界补齐位置。
        scores = tl.where(key_indices[None, :] < N_KEYS, scores, float("-inf"))
        if is_causal:
            allowed = query_indices[:, None] >= key_indices[None, :]
            # 讲义要求：因果 mask 给分数加 -1e6。
            scores = scores + tl.where(allowed, 0.0, -1e6)

        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m, block_max)
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new[:, None])
        l = alpha * l + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc=acc, input_precision="ieee")
        m = m_new
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    output = acc / l[:, None]
    logsumexp = m + tl.log(l)
    tl.store(
        O_block_ptr, output.to(O_ptr.dtype.element_ty), boundary_check=(0, 1)
    )
    tl.store(L_block_ptr, logsumexp, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        validate_inputs(Q, K, V)
        if not Q.is_cuda:
            raise ValueError("Triton 版本需要 CUDA 张量")
        if Q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Triton 版本支持 FP16、BF16 和 FP32")

        batch_shape = Q.shape[:-2]
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        d = Q.shape[-1]
        batch_size = math.prod(batch_shape)

        # 将前导 batch 维度展平，连续输入不需要复制。
        q = Q.contiguous().view(batch_size, n_queries, d)
        k = K.contiguous().view(batch_size, n_keys, d)
        v = V.contiguous().view(batch_size, n_keys, d)
        o = torch.empty((batch_size, n_queries, d), device=Q.device, dtype=Q.dtype)
        l = torch.empty((batch_size, n_queries), device=Q.device, dtype=torch.float32)

        query_tile_size = 16
        key_tile_size = 16
        d_block = max(16, triton.next_power_of_2(d))
        grid = (triton.cdiv(n_queries, query_tile_size), batch_size)

        with torch.cuda.device(Q.device):
            flash_fwd_kernel[grid](
                q, k, v, o, l,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                o.stride(0), o.stride(1), o.stride(2),
                l.stride(0), l.stride(1),
                N_QUERIES=n_queries,
                N_KEYS=n_keys,
                scale=1.0 / math.sqrt(d),
                D=d,
                D_BLOCK=d_block,
                Q_TILE_SIZE=query_tile_size,
                K_TILE_SIZE=key_tile_size,
                is_causal=is_causal,
                num_warps=4,
            )

        O = o.view(*batch_shape, n_queries, d)
        L = l.view(*batch_shape, n_queries)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_backward_pytorch(
            Q, K, V, O, grad_output, L, ctx.is_causal
        )
        gradients = (dQ, dK, dV, None)
        return gradients[:len(ctx.needs_input_grad)]
