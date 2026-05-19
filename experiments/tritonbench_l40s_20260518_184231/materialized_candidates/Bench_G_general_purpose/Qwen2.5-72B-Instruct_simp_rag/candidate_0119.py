import triton
import triton.language as tl

@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q, k, v, g, h, do, dq, dk, dg,
    s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d,
    B, H, T, D, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    USE_G: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    o_i = tl.arange(0, BT)
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b) * scale, tl.math.exp2((BT - o_i - 1) * b_b)
    d_b = tl.math.exp2(BT * b_b)

    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * scale
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg + i_bh * DK, (DK,), (s_qk_d,), (i_k * BK,), (BK,), (0,))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dd = (b_do * d_q[:, None]).to(b_do.dtype)

        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = (b_ds * d_s).to(b_k.dtype)
        b_dq = tl.dot(b_ds, b_k, allow_tf32=False)
        b_dk = tl.dot(tl.trans(b_ds), b_v, allow_tf32=False)

        if USE_G:
            b_g = tl.load(g + i_bh * DK, boundary_check=(0,))
            b_dg = tl.dot(b_ds, b_v, allow_tf32=False) * b_g
            b_dq += b_dg * b_k
            b_dk += b_dg * b_v

        if CHECK and i == 0:
            b_dq += tl.dot(b_dd, b_h.to(b_k.dtype), allow_tf32=False)
            b_h = d_b * b_h + tl.dot((b_v * d_k[None, :]).to(b_k.dtype), b_k, allow_tf32=False)
        else:
            b_dq += tl.dot(b_dd, b_h.to(b_k.dtype), allow_tf32=False)
            b_h = d_b * b_h + tl.dot((b_v * d_k[None, :]).to(b_k.dtype), b_k, allow_tf32=False)

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        if USE_G:
            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

    b_h = None
    tl.debug_barrier()
    d_s = tl.trans(d_s)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dd = (b_do * d_q[:, None]).to(b_do.dtype)

        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = (b_ds * d_s).to(b_k.dtype)

        b_s = tl.dot(b_k, b_q, allow_tf32=False) * d_s
        b_dk = tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        b_dv = tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        if CHECK and i == 1:
            b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False) * d_k[:, None]
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False) * d_k[:, None]
            b_dh = d_b * b_dh + tl.dot(b_q, b_dd, allow_tf32=False)
        else:
            b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False) * d_k[:, None]
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False) * d_k[:, None]
            b_dh = d_b * b_dh + tl.dot(b_q, b_dd, allow_tf32=False)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty
