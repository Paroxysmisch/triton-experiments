import torch
import triton
import triton.language as tl

@triton.jit
def fused_recurrent_retention_fwd_kernel(
    q, k, v, o, h, initial_state, final_state,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    s_h_h, s_h_t, s_h_d,
    T, scale,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % HEAD_DIM

    o_i = tl.arange(0, BV)
    o_j = i_k * BV + tl.arange(0, BV)
    k_ptrs = k + i_bh * s_qk_h + (o_i[:, None] * s_qk_d + o_j[None, :] + i_v * BK)
    v_ptrs = v + i_bh * s_vo_h + (o_i[:, None] * s_vo_d + o_j[None, :] + i_v * BV)

    if USE_INITIAL_STATE:
        h_ptrs = initial_state + i_bh * s_h_h + i_h * s_h_t + tl.arange(0, BK)[None, :]
        h_i = tl.arange(0, BK)
        h_j = i_v * BK + tl.arange(0, BK)[:, None]
        h_mask = h_j < BK
        h = tl.load(h_ptrs, mask=h_mask, other=0).to(tl.float32)

    mask = tl.where(i_h == 0, (i_v * BK + tl.arange(0, BK)) < BK, tl.full([BK], 1, tl.int8))
    k = tl.load(k_ptrs, mask=mask, other=0).to(tl.float32)
    v = tl.load(v_ptrs, mask=mask, other=0).to(tl.float32)
    k = k / scale

    m = tl.zeros([BK, BK], dtype=tl.float32)
    for _ in range(0, T):
        m_i = tl.arange(0, BK)
        m_j = tl.arange(0, BK)
        m_mask = m_i[:, None] >= m_j[None, :]
        m_k = tl.dot(k, tl.trans(k))
        m = m * (1 - m_mask) + m_k * m_mask
        m_diag = tl.diag(m)
        m_diag = tl.where(m_diag <= 0, 1e8, m_diag)
        m_p = tl.where(m_mask, m, 0)
        p = m_p / tl.sqrt(m_diag[:, None] * m_diag[None, :])
        h = tl.dot(p, h) + tl.dot(k, v)

        if STORE_FINAL_STATE:
            final_state_ptrs = final_state + i_bh * s_h_h + i_h * s_h_t + tl.arange(0, BK)[None, :]
            final_state_mask = final_state_ptrs + (i_h * s_h_t + tl.arange(0, BK)[:, None]) < final_state + i_bh * s_h_h + i_h * s_h_t + BK * s_h_d
            tl.store(final_state_ptrs, h, mask=final_state_mask)

        k = k * 0.99
        v = v * 0.99
        mask = mask * 0

    o_ptrs = o + i_bh * s_vo_h + (o_i[:, None] * s_vo_d + o_j[None, :] + i_v * BV)
    o_mask = o_ptrs + (i_h * s_vo_t + tl.arange(0, BV)[:, None]) < o + i_bh * s_vo_h + i_h * s_vo_t + BV * s_vo_d
    tl.store(o_ptrs, h, mask=o_mask)

@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q, k, v, do, dq, dk, dv, dh,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    s_h_h, s_h_t, s_h_d,
    T, scale,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % HEAD_DIM

    o_i = tl.arange(0, BV)
    o_j = i_k * BV + tl.arange(0, BV)
    k_ptrs = k + i_bh * s_qk_h + (o_i[:, None] * s_qk_d + o_j[None, :] + i_v * BK)
    v_ptrs = v + i_bh * s_vo_h + (o_i[:, None] * s_vo_d + o_j[None, :] + i_v * BV)
    do_ptrs = do + i_bh * s_vo_h + (o_i[:, None] * s_vo_d + o_j[None, :] + i_v * BV)

    mask = tl.where(i_h == 0, (i_v * BK + tl.arange(0, BK)) < BK, tl.full([BK], 1, tl.int8))
    k = tl.load(k_ptrs, mask=mask, other=0).to(tl.float32)
    v = tl.load(v_ptrs, mask=mask, other=0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask, other=0).to(tl.float32)
    k = k / scale

    dq = tl.zeros([BK, BV], dtype=tl.float32)
    dk = tl.zeros([BK, BK], dtype=tl.float32)
    dv = tl.zeros([BK, BV], dtype=tl.float32)
    for t in range(T - 1, -1, -1):
        dq_i = tl.arange(0, BK)
        dq_j = tl.arange(0, BV)
        dq_mask = dq_i[:, None] < BK
        dq_p = tl.dot(do, tl.trans(k))
        dq = dq * (1 - dq_mask) + dq_p * dq_mask
        dq_diag = tl
