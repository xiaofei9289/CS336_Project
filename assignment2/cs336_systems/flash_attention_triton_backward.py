"""CS336 4.2.3: optional tiled Triton backward (handout Algorithm 2).

Place this NEW file beside the existing cs336_systems/flash_attention.py.
The required PyTorch+torch.compile backward remains in that original file.

Public API:
    FlashAttentionTritonFull.apply(Q, K, V, is_causal=False)
    flash_backward_triton(Q, K, V, O, dO, L, is_causal=False)

Forward is inherited from the existing FlashAttentionTriton. Backward uses:
    1. A row-wise D = sum(O * dO) preprocessing kernel.
    2. A kernel owning each key tile, computing dK and dV.
    3. A kernel owning each query tile, computing dQ.

No atomic additions, dense NxN buffers, PyTorch matmuls, or compiled PyTorch
backward calls. The two gradient kernels each recompute their own P tiles.
All backward dot operands and accumulators use FP32 with IEEE input precision.
This is a correctness-first implementation; faster performance is not assumed.
Only first-order gradients are supported.
"""

import math

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from cs336_systems.flash_attention import FlashAttentionTriton


# ============================================================
# 1. 预计算 D：每个 query 行只有一个 FP32 标量。
#    Python 包装保证传入内核的张量连续，因此使用简单的偏移计算。
# ============================================================

@triton.jit
def _flash_bwd_delta_kernel(
    O_ptr,
    dO_ptr,
    Delta_ptr,
    N_QUERIES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_BLOCK: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    query_indices = (
        query_tile_index * Q_TILE_SIZE
        + tl.arange(0, Q_TILE_SIZE)
    )
    dim_indices = tl.arange(0, D_BLOCK)

    offsets = (
        batch_index * N_QUERIES * HEAD_DIM
        + query_indices[:, None] * HEAD_DIM
        + dim_indices[None, :]
    )
    valid = (
        (query_indices[:, None] < N_QUERIES)
        & (dim_indices[None, :] < HEAD_DIM)
    )

    o = tl.load(O_ptr + offsets, mask=valid, other=0).to(tl.float32)
    do = tl.load(dO_ptr + offsets, mask=valid, other=0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    tl.store(
        Delta_ptr + batch_index * N_QUERIES + query_indices,
        delta,
        mask=query_indices < N_QUERIES,
    )


# ============================================================
# 2. 每个程序实例固定一个 key tile，遍历所有 query tiles。
#    当前实例独占对应的 dK/dV 输出块，无需 atomic_add。
# ============================================================

@triton.jit
def _flash_bwd_dkdv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    Delta_ptr,
    dK_ptr,
    dV_ptr,
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_BLOCK: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    scale: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    key_indices = (
        key_tile_index * K_TILE_SIZE
        + tl.arange(0, K_TILE_SIZE)
    )
    query_offsets = tl.arange(0, Q_TILE_SIZE)
    dim_indices = tl.arange(0, D_BLOCK)

    kv_offsets = (
        batch_index * N_KEYS * HEAD_DIM
        + key_indices[:, None] * HEAD_DIM
        + dim_indices[None, :]
    )
    kv_valid = (
        (key_indices[:, None] < N_KEYS)
        & (dim_indices[None, :] < HEAD_DIM)
    )

    # K、V 在本实例中固定，只加载一次。
    k = tl.load(K_ptr + kv_offsets, mask=kv_valid, other=0).to(tl.float32)
    v = tl.load(V_ptr + kv_offsets, mask=kv_valid, other=0).to(tl.float32)

    dk_acc = tl.zeros((K_TILE_SIZE, D_BLOCK), dtype=tl.float32)
    dv_acc = tl.zeros((K_TILE_SIZE, D_BLOCK), dtype=tl.float32)

    for query_tile in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        query_indices = query_tile * Q_TILE_SIZE + query_offsets
        q_offsets = (
            batch_index * N_QUERIES * HEAD_DIM
            + query_indices[:, None] * HEAD_DIM
            + dim_indices[None, :]
        )
        q_valid = (
            (query_indices[:, None] < N_QUERIES)
            & (dim_indices[None, :] < HEAD_DIM)
        )

        q = tl.load(Q_ptr + q_offsets, mask=q_valid, other=0).to(tl.float32)
        do = tl.load(dO_ptr + q_offsets, mask=q_valid, other=0).to(tl.float32)

        row_offsets = batch_index * N_QUERIES + query_indices
        logsumexp = tl.load(
            L_ptr + row_offsets,
            mask=query_indices < N_QUERIES,
            other=0,
        ).to(tl.float32)
        delta = tl.load(
            Delta_ptr + row_offsets,
            mask=query_indices < N_QUERIES,
            other=0,
        )

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale

        if is_causal:
            allowed = query_indices[:, None] >= key_indices[None, :]
            # 与已有前向一致：因果位置加 -1e6。
            scores = scores + tl.where(allowed, 0.0, -1e6)

        # 越界 query/key 的概率必须为零；-inf 仅用于边界补齐。
        score_valid = (
            (query_indices[:, None] < N_QUERIES)
            & (key_indices[None, :] < N_KEYS)
        )
        scores = tl.where(score_valid, scores, float("-inf"))
        p = tl.exp(scores - logsumexp[:, None])

        # dV += P^T @ dO
        dv_acc = tl.dot(
            tl.trans(p), do,
            acc=dv_acc,
            input_precision="ieee",
        )

        # dS = P * (dO @ V^T - D)
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - delta[:, None])

        # 先累加，最后统一乘 scale。
        dk_acc = tl.dot(
            tl.trans(ds), q,
            acc=dk_acc,
            input_precision="ieee",
        )

    tl.store(
        dK_ptr + kv_offsets,
        (dk_acc * scale).to(dK_ptr.dtype.element_ty),
        mask=kv_valid,
    )
    tl.store(
        dV_ptr + kv_offsets,
        dv_acc.to(dV_ptr.dtype.element_ty),
        mask=kv_valid,
    )


# ============================================================
# 3. 每个程序实例固定一个 query tile，遍历所有 key tiles。
#    重新计算 P，当前实例独占对应的 dQ 输出块。
# ============================================================

@triton.jit
def _flash_bwd_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    Delta_ptr,
    dQ_ptr,
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_BLOCK: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    scale: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    query_indices = (
        query_tile_index * Q_TILE_SIZE
        + tl.arange(0, Q_TILE_SIZE)
    )
    key_offsets = tl.arange(0, K_TILE_SIZE)
    dim_indices = tl.arange(0, D_BLOCK)

    q_offsets = (
        batch_index * N_QUERIES * HEAD_DIM
        + query_indices[:, None] * HEAD_DIM
        + dim_indices[None, :]
    )
    q_valid = (
        (query_indices[:, None] < N_QUERIES)
        & (dim_indices[None, :] < HEAD_DIM)
    )

    q = tl.load(Q_ptr + q_offsets, mask=q_valid, other=0).to(tl.float32)
    do = tl.load(dO_ptr + q_offsets, mask=q_valid, other=0).to(tl.float32)

    row_offsets = batch_index * N_QUERIES + query_indices
    logsumexp = tl.load(
        L_ptr + row_offsets,
        mask=query_indices < N_QUERIES,
        other=0,
    ).to(tl.float32)
    delta = tl.load(
        Delta_ptr + row_offsets,
        mask=query_indices < N_QUERIES,
        other=0,
    )

    dq_acc = tl.zeros((Q_TILE_SIZE, D_BLOCK), dtype=tl.float32)

    for key_tile in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        key_indices = key_tile * K_TILE_SIZE + key_offsets
        kv_offsets = (
            batch_index * N_KEYS * HEAD_DIM
            + key_indices[:, None] * HEAD_DIM
            + dim_indices[None, :]
        )
        kv_valid = (
            (key_indices[:, None] < N_KEYS)
            & (dim_indices[None, :] < HEAD_DIM)
        )

        k = tl.load(K_ptr + kv_offsets, mask=kv_valid, other=0).to(tl.float32)
        v = tl.load(V_ptr + kv_offsets, mask=kv_valid, other=0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale

        if is_causal:
            allowed = query_indices[:, None] >= key_indices[None, :]
            scores = scores + tl.where(allowed, 0.0, -1e6)

        score_valid = (
            (query_indices[:, None] < N_QUERIES)
            & (key_indices[None, :] < N_KEYS)
        )
        scores = tl.where(score_valid, scores, float("-inf"))
        p = tl.exp(scores - logsumexp[:, None])

        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - delta[:, None])

        dq_acc = tl.dot(
            ds, k,
            acc=dq_acc,
            input_precision="ieee",
        )

    tl.store(
        dQ_ptr + q_offsets,
        (dq_acc * scale).to(dQ_ptr.dtype.element_ty),
        mask=q_valid,
    )


# ============================================================
# Python 只负责输入布局、分配输出和启动内核。
# ============================================================

def flash_backward_triton(
    Q,
    K,
    V,
    O,
    dO,
    L,
    is_causal=False,
    *,
    query_tile_size=32,
    key_tile_size=32,
):
    if not Q.is_cuda:
        raise ValueError("Triton backward 需要 CUDA 张量")
    if Q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Triton backward 支持 FP16、BF16 和 FP32")
    if O.shape != Q.shape or dO.shape != Q.shape:
        raise ValueError("O、dO 的形状必须与 Q 一致")
    if L.shape != Q.shape[:-1]:
        raise ValueError("L 的形状必须为 Q.shape[:-1]")
    if any(t.device != Q.device for t in (O, dO, L)):
        raise ValueError("所有张量必须位于同一设备")
    if O.dtype != Q.dtype or dO.dtype != Q.dtype or L.dtype != torch.float32:
        raise ValueError("O/dO 应与 Q 同 dtype，L 应为 FP32")
    for tile_size in (query_tile_size, key_tile_size):
        if tile_size < 16 or tile_size & (tile_size - 1):
            raise ValueError("tile size 必须是不小于 16 的 2 的幂")

    batch_shape = Q.shape[:-2]
    batch_size = math.prod(batch_shape)
    n_queries = Q.shape[-2]
    n_keys = K.shape[-2]
    head_dim = Q.shape[-1]

    # 非连续输入只产生线性大小的布局副本，不产生 NxN 张量。
    q = Q.contiguous().view(batch_size, n_queries, head_dim)
    k = K.contiguous().view(batch_size, n_keys, head_dim)
    v = V.contiguous().view(batch_size, n_keys, head_dim)
    o = O.contiguous().view(batch_size, n_queries, head_dim)
    do = dO.contiguous().view(batch_size, n_queries, head_dim)
    logsumexp = L.contiguous().view(batch_size, n_queries)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty(
        (batch_size, n_queries),
        device=Q.device,
        dtype=torch.float32,
    )

    d_block = max(16, triton.next_power_of_2(head_dim))
    query_grid = (
        triton.cdiv(n_queries, query_tile_size),
        batch_size,
    )
    key_grid = (
        triton.cdiv(n_keys, key_tile_size),
        batch_size,
    )
    common = dict(
        N_QUERIES=n_queries,
        N_KEYS=n_keys,
        HEAD_DIM=head_dim,
        D_BLOCK=d_block,
        Q_TILE_SIZE=query_tile_size,
        K_TILE_SIZE=key_tile_size,
        scale=1.0 / math.sqrt(head_dim),
        is_causal=is_causal,
        num_warps=4,
    )

    with torch.cuda.device(Q.device):
        # 同一 stream 按顺序执行，无需在内核之间手动 synchronize。
        _flash_bwd_delta_kernel[query_grid](
            o,
            do,
            delta,
            N_QUERIES=n_queries,
            HEAD_DIM=head_dim,
            D_BLOCK=d_block,
            Q_TILE_SIZE=query_tile_size,
            num_warps=4,
        )

        _flash_bwd_dkdv_kernel[key_grid](
            q,
            k,
            v,
            do,
            logsumexp,
            delta,
            dk,
            dv,
            **common,
        )

        _flash_bwd_dq_kernel[query_grid](
            q,
            k,
            v,
            do,
            logsumexp,
            delta,
            dq,
            **common,
        )

    return dq.view(Q.shape), dk.view(K.shape), dv.view(V.shape)


# ============================================================
# 复用已有 Triton 前向，仅替换 backward。
# ============================================================

class FlashAttentionTritonFull(FlashAttentionTriton):
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors

        dQ, dK, dV = flash_backward_triton(
            Q,
            K,
            V,
            O,
            grad_output,
            L,
            ctx.is_causal,
        )

        gradients = (dQ, dK, dV, None)
        return gradients[:len(ctx.needs_input_grad)]
