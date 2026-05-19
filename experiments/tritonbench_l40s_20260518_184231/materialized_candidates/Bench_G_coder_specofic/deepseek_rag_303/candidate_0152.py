import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def chunk_retention_fwd_kernel_h(
    k,
    initial_state,
    h,
    d_b, 
    T: tl.constexpr,
    E: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_e, i_h, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_h = i_h * BH + tl.arange(0, BH)
    o_e = i_e * BK + tl.arange(0, BK)
    o_d = i_h * BH + tl.arange(0, BD)
    NT = tl.cdiv(T, BT)

    mask_h = (o_h < H)
    mask_e = (o_e < BK) and (i_e * BK < E)
    mask_d = (o_d < D)

    d_i = tl.arange(0, BK)
    d_b = ((d_b * (d_i[:, None] == tl.arange(0, BK)[None, :]))).to(tl.float32)
    
    b_ktop1 = (i_e + 1) * BK
    mask_k = mask_e & (o_e < b_ktop1)

    b_h = tl.zeros([BH, BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        m_i = (i_bh == i_e) & mask_h
        b_h += tl.load(initial_state + i_bh * H + o_h, mask=m_i[:, None], other=0).to(tl.float32)

    for i_t in range(NT):
        p_k = (k + (i_bh + i_t * BH * E) * D + i_e * BK + o_e)
        p_h = (h + (i_bh + i_t * BH * E) * D + o_h * D + o_d)

        b_g = tl.load(p_k, mask=mask_k[:, None] & mask_d[None, :], other=0).to(tl.float32) * d_b[:, :, None]
        b_h = b_h * (1 - tl.sum(b_g, axis=1)[:, None]) + b_g
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=(mask_h[:, None] & mask_e[None, :]) & mask_k)

    if STORE_FINAL_STATE:
        m_e = (i_e == E-1) & (i_bh == i_h) & mask_h
        p_s = h + (i_bh + i_t * BH * E) * D + o_h * D + o_d
        final_state = tl.load(p_s, mask=m_e[:, None], other=0)
        tl.store(p_s, final_state.to(p_s.dtype.element_ty), mask=m_e[:, None])


@triton.jit
def chunk_retention_fwd_kernel_o(q, k, v, h, o, s_h, s_k, s_v, s_e, s_d, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    i_d, i_e, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_e = i_e * BK + tl.arange(0, BK)
    o_d = i_d * BD + tl.arange(0, BD)

    NT = tl.cdiv(T, BT)
    mask_e = (o_e < BK) & (i_e * BK < E)
    mask_d = (o_d < D)

    mask_h = ((NT - 1) * BT + tl.arange(0, BK)) < T
    d_b = (TL_MAX_FLT / (1 + tl.exp((((BT - tl.arange(0, BK) * 1.) / BT) * 8))))
    d_b = ((d_b * (o_e[:, None] == tl.arange(0, BK)[None, :]))).to(tl.float32)

    b_q = tl.zeros([BK, BD], dtype=tl.float32)
    b_s = tl.zeros([BK], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_h, (T, E, D), (s_k, s_v, s_d), (i_t * BT, i_e * BK, i_d * BD), (BT, BK, BD), (1, 0, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_h, (T, E, D), (s_k, s_v, s_d), (i_t * BT, i_e * BK, i_d * BD), (BT, BK, BD), (1, 0, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_h, (T, E, D), (s_k, s_v, s_d), (i_t * BT, i_e * BK, i_d * BD), (BT, BK, BD), (1, 0, 0))
        p_o = tl.make_block_ptr(o + i_bh * s_h, (T, E, D), (s_k, s_v, s_d), (i_t * BT, i_e * BK, i_d * BD), (BT, BK, BD), (1, 0, 0))
        p_h0 = tl.make_block_ptr(h + i_bh * s_h, (T, E, D), (s_k, s_v, s_d), (i_t * BT - 1, i_e * BK, i_d * BD), (BT, BK, BD), (1, 0, 0))

        b_e = tl.load(p_k, boundary_check=(0, 1, 2))
        b_q += tl.sum(b_e.to(tl.float32) * tl.load(p_q, boundary_check=(0, 1, 2)), axis=1)[:, None]
        if mask_h[i_t]:
            b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
            b_o = tl.sum((b_v * d_b * b_h).to(b_o.dtype.element_ty), axis=1).to(tl.float32)
            tl.store(p_o, b_o, boundary_check=(0, 1))

        else:
            tl.store(p_o, b_h0, boundary_check=(0, 1))
