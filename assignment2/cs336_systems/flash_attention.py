"""CS336 Assignment 2 FlashAttention-2.

Q, K, and V have shape (batch, seq, d). Both sequence lengths must be
divisible by 16; integer division drops any leftover rows. O matches Q's
dtype, and L is the FP32 logsumexp. Masked scores receive -1e6. Both
classes save (L, Q, K, V, O) and return only O.
"""

import math

import torch
import triton
import triton.language as tl


Q_TILE_SIZE = 16
K_TILE_SIZE = 16


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        scale = 1.0 / math.sqrt(d)
        T_q = n_queries // Q_TILE_SIZE
        T_k = n_keys // K_TILE_SIZE

        O = torch.empty_like(Q)
        L = torch.empty(batch_size, n_queries, device=Q.device, dtype=torch.float32)

        for b in range(batch_size):
            for i in range(T_q):
                q_start = i * Q_TILE_SIZE
                q_end = q_start + Q_TILE_SIZE
                Q_i = Q[b, q_start:q_end].float()
                # Algorithm 1 initial values. Inside the loop, O_i and l_i are running
                # state rescaled to the current row max, not the final output.
                m_i = torch.full((Q_TILE_SIZE,), float("-inf"), device=Q.device)
                l_i = torch.zeros(Q_TILE_SIZE, device=Q.device)
                O_i = torch.zeros(Q_TILE_SIZE, d, device=Q.device)

                for j in range(T_k):
                    k_start = j * K_TILE_SIZE
                    k_end = k_start + K_TILE_SIZE
                    K_j = K[b, k_start:k_end].float()
                    V_j = V[b, k_start:k_end].float()
                    S_ij = (Q_i @ K_j.transpose(-2, -1)) * scale
                    if is_causal:
                        query_indices = q_start + torch.arange(Q_TILE_SIZE, device=Q.device)
                        key_indices = k_start + torch.arange(K_TILE_SIZE, device=Q.device)
                        allowed = query_indices[:, None] >= key_indices[None, :]
                        S_ij = S_ij + torch.where(allowed, 0.0, -1e6)

                    m_ij = torch.maximum(m_i, S_ij.max(dim=-1).values)
                    P_ij = torch.exp(S_ij - m_ij.unsqueeze(-1))
                    # When this tile raises the row max, alpha rescales the old l_i and O_i onto that max.
                    alpha = torch.exp(m_i - m_ij)
                    l_i = alpha * l_i + P_ij.sum(dim=-1)
                    O_i = alpha.unsqueeze(-1) * O_i + P_ij @ V_j
                    m_i = m_ij

                # Normalize only after every key tile has been accumulated.
                O[b, q_start:q_end] = (O_i / l_i.unsqueeze(-1)).to(Q.dtype)
                L[b, q_start:q_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_backward_compiled(Q, K, V, O, grad_output, L, ctx.is_causal)
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    q_start = query_tile_index * Q_TILE_SIZE

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    # Same as the PyTorch forward: O_i and l_i stay running state until O_i / l_i is stored.
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    query_indices = q_start + tl.arange(0, Q_TILE_SIZE)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        S_ij = tl.dot(Q_i, tl.trans(K_j), input_precision="ieee") * scale
        if is_causal:
            key_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            allowed = query_indices[:, None] >= key_indices[None, :]
            S_ij = S_ij + tl.where(allowed, 0.0, -1e6)

        m_ij = tl.maximum(m_i, tl.max(S_ij, axis=1))
        P_ij = tl.exp(S_ij - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = alpha * l_i + tl.sum(P_ij, axis=1)
        O_i = O_i * alpha[:, None]
        O_i = tl.dot(P_ij.to(V_j.dtype), V_j, acc=O_i, input_precision="ieee")
        m_i = m_ij
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(
        O_block_ptr,
        (O_i / l_i[:, None]).to(O_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(L_block_ptr, m_i + tl.log(l_i), boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        O = torch.empty_like(Q)
        L = torch.empty(batch_size, n_queries, device=Q.device, dtype=torch.float32)

        flash_fwd_kernel[(triton.cdiv(n_queries, Q_TILE_SIZE), batch_size)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            1.0 / math.sqrt(d),
            d,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_backward_compiled(Q, K, V, O, grad_output, L, ctx.is_causal)
        return dQ, dK, dV, None


def flash_backward_pytorch(Q, K, V, O, dO, L, is_causal):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    q = Q.float()
    k = K.float()
    v = V.float()
    o = O.float()
    do = dO.float()
    logsumexp = L.float()

    D = (o * do).sum(dim=-1)
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


flash_backward_compiled = torch.compile(flash_backward_pytorch)
