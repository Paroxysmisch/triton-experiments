import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd
from typing import Optional, Tuple

@triton.jit
def fused_recurrent_fwd_kernel(
    q, k, v, h0, ht, o, s_k_h, s_v_h,
    beta, scale, B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    HAS_BETA_SCALE: tl.constexpr, REVERSE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V

    h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_h0, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32)
        if HAS_BETA_SCALE:
            b_q = (b_q * beta[i_bh, i_k] * scale).to(tl.float32)
        b_qk = b_q * tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        h += b_qk.to(tl.float32)
        b_o = h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)
        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_o += -V if REVERSE else V
        p_v += -V if REVERSE else V

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_ht, h.to(p_ht.dtype.element_ty), mask=mask_bk[:, None] & mask_bv[None, :])


@triton.jit
def fused_recurrent_bwd_kernel(
    q, k, v, do, dq, dk, dv, h0, beta, scale, dbeta, dscale,
    s_k_h, s_v_h, B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, HAS_BETA_SCALE: tl.constexpr, REVERSE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    p_dq = dq + (i_bh + i_k * B * H) * s_k_h + i_v * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_dk = dk + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_dv = dv + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    dh = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        dh += tl.load(p_h0, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32) * beta[i_bh, i_k]

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        if USE_INITIAL_STATE:
            dh += b_k[None, :] * tl.load
