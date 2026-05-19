import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, O, L, M, 
    stride_qb, stride_qh, stride_qm, stride_qk, 
    stride_kb, stride_kh, stride_km, stride_kk, 
    stride_vb, stride_vh, stride_vk, stride_vn, 
    stride_ob, stride_oh, stride_om, stride_on, 
    stride_lb, stride_lh, stride_lm, 
    stride_mb, stride_mh, stride_mm, 
    nheads, seqlen, seqlen_rounded, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr, 
    IS_CAUSAL: tl.constexpr
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
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_km + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vk + offs_d[None, :])
    # initialize pointers to O, L, M
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    l_ptrs = L + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm
    m_ptrs = M + off_b * stride_mb + off_h * stride_mh + offs_m * stride_mm

    # initialize block-wide accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    l_max = tl.zeros((BLOCK_M,), dtype=tl.float32)
    l_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # loop over k
    for start_n in range(0, seqlen_rounded, BLOCK_N):
        # -- compute qk ----
        k_block_ptr = k_ptrs + start_n * stride_km
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        q = tl.load(q_ptrs)
        k = tl.load(k_block_ptr)
        qk += tl.dot(q, k, trans_b=True) * sm_scale
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float('-inf'))
        # -- compute l_max --
        l_max_new = tl.max(qk, 1)
        l_max = tl.where(start_n == 0, l_max_new, tl.maximum(l_max, l_max_new))
        # -- compute exp(qk - l_max) --
        qk = tl.exp(qk - l_max[:, None])
        # -- compute l_sum --
        l_sum_new = tl.sum(qk, 1)
        l_sum = tl.where(start_n == 0, l_sum_new, l_sum + l_sum_new)
        # -- update acc --
        v_block_ptr = v_ptrs + start_n * stride_vk
        v = tl.load(v_block_ptr)
        acc += tl.dot(qk.to(tl.float16), v)
        # -- update l and m --
        tl.store(l_ptrs, l_sum)
        tl.store(m_ptrs, l_max)

    # -- write back O --
    acc = acc / l_sum[:, None]
    tl.store(o_ptrs, acc.to(tl.float16))

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DMODEL': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_DMODEL': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_DMODEL': 64}, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_DMODEL': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_DMODEL': 64}, num_warps=4),
    ],
    key=['seqlen', 'nheads', 'dmodel'],
)
@triton.heuristics(
    {
        'Lk': lambda args: min(128, args['dmodel']),
    }
)
@triton.jit
def flash_attn_triton(
    q, k, v, sm_scale, o, l, m, 
    seqlen, nheads, dmodel, 
    IS_CAUSAL: tl.constexpr
):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = dmodel
    seqlen_rounded = (seqlen + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    grid = (triton.cdiv(seqlen, BLOCK_M), nheads * q.shape[0])
    _fwd_kernel[grid](
        q, k, v, sm_scale, o, l, m, 
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), 
        k.stride(0), k.stride(1), k.stride(2), k.stride(3), 
        v.stride(0), v.stride(1), v.stride(2), v.stride(3), 
        o.stride(0), o.stride(1), o.stride(2), o.stride(3), 
        l.stride(0), l.stride(1), l.stride(2), 
        m.stride(0), m.stride(1), m.stride(2), 
        nheads, seqlen, seqlen_rounded, 
        BLOCK_M, BLOCK_N, BLOCK_DMODEL, 
        IS_CAUSAL
    )

# Example usage
batch_size = 2
nheads = 4
seqlen = 128
dmodel = 64
q = torch.randn((batch_size, nheads, seqlen, dmodel), device='cuda', dtype=torch.float16)
k = torch.randn((batch_size, nheads, seqlen, dmodel), device='cuda', dtype=torch.float16)
v = torch.randn((batch_size, nheads, seqlen, dmodel), device='cuda', dtype=torch.float16)
o = torch.empty_like(q)
l = torch.empty((batch_size, nheads, seqlen), device='cuda', dtype=torch.float32)
m = torch.empty((batch_size, nheads, seqlen), device='cuda', dtype=torch.float32)
sm_scale = 1.0 / (dmodel ** 0.5)
flash_attn_triton(q, k, v, sm_scale, o, l, m, seqlen, nheads, dmodel, IS_CAUSAL=True)
