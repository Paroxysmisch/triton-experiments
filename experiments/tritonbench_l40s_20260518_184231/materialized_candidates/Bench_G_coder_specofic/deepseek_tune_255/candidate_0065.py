import torch
import triton
import triton.language as tl
from torch.autograd.function import Function

@triton.jit
def fused_recurrent_fwd_kernel(
    q, k, v, beta, initial_state, o,
    s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d,
    B, H, T, K, V, BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, BETA_SCALE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_k * BK), (BK, BV), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BV), (0, 1) if i_k % 2 == 0 else (0, -1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BK, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BK, BV), (1, 0))

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    mask = (i_k * BK + tl.arange(0, BK)[:, None]) < K

    for i in range(0, tl.cdiv(T, BK)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_o = tl.zeros([BK, BV], dtype=tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if BETA_SCALE:
            b_beta = tl.load(beta + i_bh * K * V + i_k * BK + tl.arange(0, BK)[:, None] * V + i_v * BV + tl.arange(0, BV)[None, :]).to(tl.float32)
            b_q = b_q * b_beta
        b_h = tl.dot(b_q.to(b_k.dtype), b_k, allow_tf32=False)
        b_h = tl.where(mask, b_h, 0)
        b_o = b_o + b_h
        b_o = b_o.to(b_v.dtype)
        b_o = tl.dot(b_v, b_o.to(b_v.dtype), allow_tf32=False)
        if IS_CAUSAL:
            b_o = tl.where(
                tl.arange(0, BK)[:, None] + i * BK >= tl.arange(0, BK)[None, :] + tl.cdiv(T, BK) * BK,
                b_o, 0
            )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BK, 0))
        p_k = tl.advance(p_k, (0, BK)) if i_k % 2 == 0 else tl.advance(p_k, (0, -BK))
        p_v = tl.advance(p_v, (BK, 0))
        p_o = tl.advance(p_o, (BK, 0))

    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(o + (i_bh + i_k * B * H + T - 1) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_o.to(p_final.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def fused_recurrent_bwd_kernel(
    q, k, v, beta, initial_state, do, dq, dk, dv, dbeta,
    s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d,
    B, H, T, K, V, BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, BETA_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BK, BV), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_k * BK), (BK, BV), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BV), (0, 1) if i_k % 2 == 0 else (0, -1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BK, BV), (1, 0))
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
    if BETA_SCALE:
