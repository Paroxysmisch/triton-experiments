import torch
import triton
import triton.language as tl

@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q, k, v, h, g, do, dh,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    B, H, T, V, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    USE_G: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    o_i = tl.arange(0, BT)
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_b = tl.math.exp2(BT * b_b)
    d_o = tl.math.exp2((o_i + 1) * b_b)
    d_h = tl.math.exp2((BT - o_i - 1) * b_b)

    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_dh = tl.make_block_ptr(dh + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dg = tl.zeros([BT, BK], dtype=tl.float32)

    NT = tl.cdiv(T, BT)
    for i in range(0, NT):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_k.dtype)

        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s
        b_o = tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        b_h = d_b * b_h + tl.dot(b_k, (b_v * d_h[:, None]).to(b_k.dtype), allow_tf32=False)

        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = (b_ds * d_s).to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
        b_dk += tl.dot(tl.trans(b_ds), b_q, allow_tf32=False)

        if USE_G:
            b_dg += b_do * b_g * scale

        if CHECK and i == 0:
            b_dq += tl.dot(b_do, b_h.to(b_k.dtype), allow_tf32=False)
            b_h = d_b * b_h + tl.dot(b_k, (b_v * d_h[:, None]).to(b_k.dtype), allow_tf32=False)
        else:
            b_dq += tl.dot(b_do, b_h.to(b_k.dtype), allow_tf32=False)
            b_h = d_b * b_h + tl.dot(b_k, (b_v * d_h[:, None]).to(b_k.dtype), allow_tf32=False)

        tl.store(p_dq, b_dq.to(p_q.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_k.dtype.element_ty), boundary_check=(0, 1))
        if USE_G:
            tl.store(p_dg, b_dg.to(p_g.dtype.element_ty), boundary_check=(0, 1))

        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_h = tl.advance(p_h, (BT, 0))
        p_g = tl.advance(p_g, (BT, 0))
        p_do = tl.advance(p_do, (BT, 0))
        p_dh = tl.advance(p_dh, (BT, 0))

@triton.autotune(
    configs=[
        triton.Config({'BT': 64, 'BK': 64, 'BV': 64}, num_warps=4),
        triton.Config({'BT': 64, 'BK': 64, 'BV': 64}, num_warps=8),
    ],
    key=['B', 'H', 'T', 'V'],
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q, k, v, h, g, do, dh,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    B, H, T, V, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    USE_G: tl.constexpr,
    CHECK: tl.constexpr
):
    chunk_simple_gla_bwd_kernel_dqkg[q, k, v, h, g, do, dh, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, V, scale, BT, BK, BV, DK, DV, USE_G, CHECK]
