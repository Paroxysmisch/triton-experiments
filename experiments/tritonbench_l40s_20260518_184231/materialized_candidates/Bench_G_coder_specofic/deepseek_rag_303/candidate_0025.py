import torch
from torch.cuda.amp import custom_bwd, custom_fwd
import triton
from triton import language as tl

@triton.jit
def _fwd_kernel(
        Q, K, V, Bias, Out,
        Lse, TMP,
        sm_scale,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn,
        stride_bb, stride_bh, stride_bm,
        stride_ob, stride_oh, stride_om,
        nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
        Lk_div_128,
        IS_CAUSAL: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        num_stages: tl.constexpr,  # number of passes that each fused kernel is split into,
):
    start_m, start_n, start_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    offs_m = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + (start_bh // nheads) * stride_qb + (start_bh % nheads) * stride_qh + \
             (offs_m[:, None] * stride_qm + offs_d[None, :])

    k_ptrs = K + (start_bh // nheads) * stride_kb + (start_bh % nheads) * stride_kh + \
             (start_n * stride_kn + offs_d[None, :])

    v_ptrs = V + (start_bh // nheads) * stride_vb + (start_bh % nheads) * stride_vh + \
             (start_n * stride_vn + offs_d[None, :])

    if Lk_div_128 == 0:
        b_ptrs = Bias + (start_bh // nheads) * stride_bb + (start_bh % nheads) * stride_bh + start_n + \
                 tl.arange(0, BLOCK_SIZE_N)

    load_mask_qk = start_m < seqlen_q
    load_mask_qd = offs_d < headdim
    load_mask_kv = start_n < seqlen_k
    load_mask_ov = load_mask_qk
    if IS_CAUSAL:
        causal_mask = offs_m[:, None] >= (start_n + offs_d)[None, :]

    # initial accumulator with float32(-inf) to provide correct reduction
    acc_o = tl.zeros([BLOCK_SIZE_M, BLOCK_HEADDIM], dtype=tl.float32) - float("inf")
    lse_i = tl.zeros([BLOCK_SIZE_M, num_stages], dtype=tl.float32) - float("inf")

    # loop through kv dimension
    for start_nd in range(0, start_n + BLOCK_SIZE_N, BLOCK_SIZE_N):
        offs_n = start_nd + tl.arange(0, BLOCK_SIZE_N)
        if Lk_div_128 == 1:
            _b_ptrs = Bias + (start_bh // nheads) * stride_bb + \
                      (start_bh % nheads) * stride_bh + offs_m[:, None] + offs_n[None, :]
        q = tl.load(q_ptrs, mask=load_mask_qk, other=float("0"))
        if Lk_div_128 == 1:
            offsets_bn = (tl.arange(0, BLOCK_SIZE_N) + start_nd)[None, :]
            if (offs_n[None, :] < seqlen_k) & (offs_m[:, None] < seqlen_q):
                bias = tl.load(_b_ptrs, mask=offs_n[None, :] < seqlen_k, other=0.0)
                bias = bias.to(q.dtype)
                q = q * bias

        # pre-mask q, k
        q = tl.where(load_mask_qk & load_mask_qd, q, 0)

        # mask by causal
        if IS_CAUSAL:
            causal_mask_q = load_mask_qk
            if causal_mask_q.dtype != tl.int32:
                causal_mask_q = causal_mask_q.to(tl.int32)
            q = tl.where(causal_mask_q[None, :] & causal_mask, 0, q)

        k = tl.load(k_ptrs, mask=load_mask_kv, other=float("0"))
        qk = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

        qk += tl.dot(q, k, trans_b=True)
        if start_n + BLOCK_SIZE_N > seqlen_k:
            qk = tl.where((start_nd + offs_n[None, :]) < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(causal_mask, qk, float("-inf"))
        if Lk_div_128 == 1:
            bias = tl.load(_b_ptrs, mask=load_mask_kv & offs_n[None, :] < seqlen_k, other=0.0)
            bias = bias.to(k.dtype)
            k = k * bias

        m_ij = tl.max(qk, 1)
        m_ij = tl.maximum(m_ij, lse_i)
        p = tl.exp((qk - m_ij[:, None]) * FLOAT32_120)
        l_ij = tl.sum(p, 1)

        # update acc_o and lse
        acc_o = acc_o * tl.exp(lse_i - m_ij[:, None])
        lse_i = m_ij + tl.log1p(l_ij)

        v = tl.load(v_ptrs, mask=load_mask_kv, other=float("0"))
        if Lk_div_128 == 1:
            bias = tl.load(_b_ptrs, mask=offs_n[None, :] < seqlen_k, other=0.0)
            bias = bias.to(v.dtype)
            v = v * bias

        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        k_ptrs += BLOCK_SIZE_N * stride_kn
        v_ptrs += BLOCK_SIZE_N * stride_vn

    lse_ptrs = Lse + (start_bh // nheads) * seqlen_q_rounded + start_m
    l_i = tl.max(lse_i) + 1024.0  # log2(0) = -Inf, but add some for numerical safety
    tl.store(lse_ptrs, l_i)

    # back to origin
    load_mask_qk = start_m < seqlen_q
    load_mask_qd = offs_d < headdim
    load_mask_kv = start_n < seqlen_k
    load_mask_ov = load_mask_qk
