import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    Q,  # Pointers to matrices
    K,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    nheads,
    rotary_dim,
    seqlen_ro,
    # strides
    stride_q_batch,
    stride_q_seqlen,
    stride_q_nheads,
    stride_q_headdim,
    stride_k_batch,
    stride_k_seqlen,
    stride_k_nheads,
    stride_k_headdim,
    stride_cos_seqlen,
    stride_cos_headdim,
    stride_sin_seqlen,
    stride_sin_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        Q = Q + pid_batch * stride_q_batch + pid_head * stride_q_nheads
        K = K + pid_batch * stride_k_batch + pid_head * stride_k_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        Q = Q + start_idx * stride_q_seqlen + pid_head * stride_q_nheads
        K = K + start_idx * stride_k_seqlen + pid_head * stride_k_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        Q = Q + (rm[:, None] * stride_q_seqlen +
                 rk_half[None, :] * stride_q_headdim)
        K = K + (rm[:, None] * stride_k_seqlen +
                 rk_half[None, :] * stride_k_headdim)
        COS = COS + (rm_cs[:, None] * stride_cos_seqlen +
                     rk_half[None, :] * stride_cos_headdim)
        SIN = SIN + (rm_cs[:, None] * stride_sin_seqlen +
                     rk_half[None, :] * stride_sin_headdim)
        cos = tl.load(
            COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0
        ).to(tl.float32)
        sin = tl.load(
            SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        q0 = tl.load(
            Q, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        q1 = tl.load(
            Q + rotary_dim_half * stride_q_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        k0 = tl.load(
            K, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        k1 = tl.load(
            K + rotary_dim_half * stride_k_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0_q = q0 * cos - q1 * sin
        o1_q = q0 * sin + q1 * cos
        o0_k = k0 * cos - k1 * sin
        o1_k = k0 * sin + k1 * cos
        Q = Q + (rm[:, None] * stride_q_seqlen +
                 rk_half[None, :] * stride_q_headdim)
        K = K + (rm[:, None] * stride_k_seqlen +
                 rk_half[None, :] * stride_k_headdim)
        tl.store(Q, o0_q, mask=(rm[:, None] < seqlen)
                 & (rk_half[None, :] < rotary_dim_half))
        tl.store(
            Q + rotary_dim_half * stride_q_headdim,
            o1_q,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
        tl.store(K, o0_k, mask=(rm[:, None] < seqlen)
                 & (rk_half[None, :] < rotary_dim_half))
        tl.store(
            K + rotary_dim_half * stride_k_headdim,
            o1_k,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        Q0 = Q + (rm[:, None] * stride_q_seqlen +
                  rk[None, :] * stride_q_headdim)
        Q1 = Q + (rm[:, None] * stride_q_seqlen +
                  rk_swap[None, :] * stride_q_headdim)
        K0 = K + (rm[:, None] * stride_k_seqlen +
                  rk[None, :] * stride_k_headdim)
        K1 = K + (rm[:, None] * stride_k_seqlen +
                  rk_swap[None, :] * stride_k_headdim)
        COS = COS + (rm_cs[:, None] * stride_cos_seqlen +
                     rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * stride_sin_seqlen +
                     rk_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (
                rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (
                rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        q0 = tl.load(Q0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(
            tl.float32
        )
        q1 = tl.load(
            Q1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        k0 = tl.load(K0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(
            tl.float32
        )
        k1 = tl.load(
            K1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        q0_cos = q0 * cos
        q1_sin = q1 * sin
        k0_cos = k0 * cos
        k1_sin = k1 * sin
        q_out = tl.where(rk[None, :] % 2 == 0, q0_cos - q1_sin, q0_cos + q1_sin)
        k_out = tl.where(rk[None, :] % 2 == 0, k0_cos - k1_sin, k0_cos + k1_sin)
        Q = Q + (rm[:, None] * stride_q_seqlen +
                 rk[None, :] * stride_q_headdim)
        K = K + (rm[:, None] * stride_k_seqlen +
