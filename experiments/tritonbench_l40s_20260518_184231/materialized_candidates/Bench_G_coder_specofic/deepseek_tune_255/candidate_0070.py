import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function
from typing import Optional, Tuple

@triton.jit
def chunk_retention_fwd_kernel_h(
    k, v, h0, h,
    s_qk_h, s_vo_h, s_h_h,
    B, H, T, scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CHECK_IN_BOUNDS: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)

    m_s = o_i[:, None] >= o_i[None, :]
    b_h0 = tl.zeros([BK, BV], dtype=tl.float32)
    p_h = h + i_bh * s_h_h
    p_k = k + i_bh * s_qk_h + (i_k * BK + o_i[:, None] * BK + o_i[None, :] * BK * DK)
    p_v = v + i_bh * s_vo_h + (i_v * BV + o_i[:, None] * BV + o_i[None, :] * BV * DV)
    p_h0 = h0 + i_bh * s_h_h + (i_k * BK + o_i[:, None] * BK + o_i[None, :] * BK * DK)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    b_h += tl.dot(b_h0.to(p_k.dtype), p_k.to(p_k.dtype).to(b_h.dtype), allow_tf32=False)
    b_h += tl.dot(p_v.to(b_h.dtype), p_k.to(p_k.dtype).to(b_h.dtype), allow_tf32=False)

    if USE_INITIAL_STATE:
        b_h += tl.load(p_h, mask=o_i[None, :] < T, other=0).to(b_h.dtype)
    tl.store(p_h + o_i[None, :] * BK + o_i[:, None] * BK * DK, b_h.to(p_h.dtype), mask=o_i[None, :] < T)

    if STORE_FINAL_STATE:
        tl.store(p_h + (T - 1) * BK + o_i[None, :] * BK + o_i[:, None] * BK * DK, b_h.to(p_h.dtype), mask=o_i[None, :] < T)
    b_h = tl.where(m_s, b_h + tl.dot(p_v.to(b_h.dtype), p_k.to(p_k.dtype).to(b_h.dtype), allow_tf32=False), b_h)
    tl.store(p_h + o_i[None, :] * BK + o_i[:, None] * BK * DK, b_h.to(p_h.dtype), mask=o_i[None, :] < T)

@triton.jit
def chunk_retention_fwd_kernel_o(
    q, k, v, h, o,
    s_qk_h, s_vo_h, s_h_h,
    B, H, T, scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    CHECK_IN_BOUNDS: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)

    m_s = o_i[:, None] >= o_i[None, :]
    p_q = q + i_bh * s_qk_h + (i_v * BV + o_i[:, None] * BV + o_i[None, :] * BV * DV)
    p_k = k + i_bh * s_qk_h + (i_k * BK + o_i[:, None] * BK + o_i[None, :] * BK * DK)
    p_v = v + i_bh * s_vo_h + (i_v * BV + o_i[:, None] * BV + o_i[None, :] * BV * DV)
    p_h = h + i_bh * s_h_h + (i_k * BK + o_i[:, None] * BK + o_i[None, :] * BK * DK)
    p_o = o + (i_bh * T + o_i[None, :] * BT + o_i[:, None] * BT * DV)
    b_o = tl.zeros([BV, BV], dtype=tl.float32)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        b_h += tl.load(p_h, mask=o_i[None, :] < T, other=0).to(b_h.dtype)
    if CHECK_IN_BOUNDS:
        b_o += tl.load(p_o, mask=m_s, other=0).to(b_o.dtype)
    else:
        b_o += tl.load(p_o).to(b_o.dtype)
    b_o += tl.dot(p_q.to(b_o.dtype), p_k.to(p_k.dtype).to(b_o.dtype), allow_tf32=False)
    b_o *= scale
    b_h += tl.dot(p_v.to(b_h.dtype), p_k.to(p_k.dtype).to(b_h.dtype), allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype), mask=m_s)
    tl.store(p_h, b_h.to(p_h.dtype), mask=o_i[None, :] < T)

@triton.jit
def chunk_retention_bwd_kernel_dh(
    q, do, dh, h,
    s_qk_h, s_vo_h, s_h_h,
    B, H, T, scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    CHECK_IN_BOUNDS: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)

    m_s = o_i
