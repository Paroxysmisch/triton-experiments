import triton
import triton.language as tl

@triton.jit
def _attn_fwd(
    Q, K, V, Out, L, M, 
    stride_qb, stride_qh, stride_qm, 
    stride_kb, stride_kh, stride_km, 
    stride_vb, stride_vh, stride_vm, 
    stride_ob, stride_oh, stride_om, 
    stride_lb, stride_lh, 
    stride_mb, stride_mh, 
    nheads, N_CTX, window_size, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # initialize pointers to Q, K, V
    q_ptr = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * 1
    k_ptr = K + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_km + offs_d[None, :] * 1
    v_ptr = V + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vm + offs_d[None, :] * 1
    # initialize pointers to output
    o_ptr = Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * 1
    # initialize pointers to L and M
    l_ptr = L + off_b * stride_lb + off_h * stride_lh + offs_m * 1
    m_ptr = M + off_b * stride_mb + off_h * stride_mh + offs_m * 1
    # initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    # load Q
    q = tl.load(q_ptr)
    # loop over k, v
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(k_ptr + start_n * stride_km)
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        # -- compute attention scores ----
        if window_size > 0:
            mask = tl.arange(0, BLOCK_N) + start_n - offs_m[:, None]
            mask = mask < window_size
            qk = tl.where(mask, qk, float('-inf'))
        m_i = tl.max(qk, 1)
        p = tl.exp(qk - m_i[:, None])
        l_i = tl.sum(p, 1)
        # -- update acc ----
        v = tl.load(v_ptr + start_n * stride_vm)
        p = p / l_i[:, None]
        acc += tl.dot(p, v)
        # -- update L and M ----
        tl.atomic_add(l_ptr + start_m * BLOCK_M, l_i)
        tl.atomic_max(m_ptr + start_m * BLOCK_M, m_i)
    # -- write back output ----
    tl.store(o_ptr, acc.to(tl.float16))

import torch
from torch import nn
from triton import cdiv

def _forward(Q, K, V, nheads, N_CTX, window_size):
    # Check inputs
    assert Q.dtype == torch.float16 and K.dtype == torch.float16 and V.dtype == torch.float16
    assert Q.shape == K.shape == V.shape
    assert Q.shape[1] == nheads
    assert Q.shape[2] == N_CTX
    assert Q.shape[3] == K.shape[3] == V.shape[3]

    # Allocate output and intermediate tensors
    Out = torch.empty_like(Q)
    L = torch.empty((Q.shape[0], nheads, N_CTX), device=Q.device, dtype=torch.float32)
    M = torch.empty((Q.shape[0], nheads, N_CTX), device=Q.device, dtype=torch.float32)

    # Define grid and block dimensions
    grid = (cdiv(N_CTX, 128), Q.shape[0] * nheads)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = Q.shape[3]

    # Launch the kernel
    _attn_fwd[grid](
        Q, K, V, Out, L, M,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        L.stride(0), L.stride(1),
        M.stride(0), M.stride(1),
        nheads, N_CTX, window_size,
        BLOCK_M, BLOCK_N, BLOCK_DMODEL
    )

    return Out, L, M
