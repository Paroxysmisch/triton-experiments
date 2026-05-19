import math
import torch
import triton
import triton.language as tl

@triton.heuristics({
    "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
    "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
})
@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    Lse, TMP,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
    CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m

    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    if EVEN_M & EVEN_N:
        q = tl.load(q_ptrs) if EVEN_HEADDIM else tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        mask_q = (offs_m[:, None] < seqlen_q)
        mask_d = (offs_d[None, :] < headdim)
        q = tl.load(q_ptrs, mask=mask_q & (mask_d if not EVEN_HEADDIM else True), other=0.0)

    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = (start_n + offs_n)[:, None] < seqlen_k
        k = tl.load(k_ptrs + start_n * stride_kn, 
                    mask=k_mask & (offs_d[None, :] < headdim) if not EVEN_HEADDIM else k_mask,
                    other=0.0)

        qk = tl.dot(q, tl.trans(k))
        qk = tl.where((start_n + offs_n)[None, :] < seqlen_k, qk, float("-inf"))
        
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None] >= (start_n + offs_n)[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])

        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o *= acc_o_scale[:, None]

        v_ptrs_n = v_ptrs + start_n * stride_vn
        v = tl.load(v_ptrs_n, 
                    mask=(start_n + offs_n)[:, None] < seqlen_k if not EVEN_N else None,
                    other=0.0)

        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)

    o_scale = tl.exp(m_i - lse_i)
    acc_o *= o_scale[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)

    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))

def flash_attn_triton(q, k, v, causal=False, sm_scale=None):
    batch, nheads, seqlen_q, d = q.shape
    _, _, seqlen_k, _ = k.shape
    assert k.shape == (batch, nheads, seqlen_k, d)
    assert v.shape == (batch, nheads, seqlen_k, d)
    assert d <= 128, "Head dimension must be ≤128"
    assert q.dtype == k.dtype == v.dtype, "Inputs must share dtype"
    assert q.is_cuda and k.is_cuda and v.is_cuda

    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(d)
    seqlen_q_rounded = (seqlen_q + 127) // 128 * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty_like(lse)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK_M = BLOCK_N = 128
    num_warps = 4 if d <= 64 else 8
    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)

    _fwd_kernel[grid](
        q, k, v, o, lse, tmp,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        nheads, seqlen_q, seqlen_k, seqlen_q_rounded, d,
        seqlen_q // 32, seqlen_k // 32,
        causal, BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=1
    )
    return o
