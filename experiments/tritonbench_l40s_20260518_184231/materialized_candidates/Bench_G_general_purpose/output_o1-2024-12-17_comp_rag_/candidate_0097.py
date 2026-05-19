import torch
import triton
import triton.language as tl

@triton.jit
def parallel_retention_fwd_kernel(
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    v,  # value [B, H, L, V]
    o,  # output [B, H, L, V]
    s_qk_h,  # stride size: L*K
    s_qk_t,  # stride size: K
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: L*V
    s_vo_t,  # stride size: V
    s_vo_d,  # stride size: 1
    scale,  # K ** -0.5
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # head count
    T: tl.constexpr,  # sequence length
    K: tl.constexpr,  # query/key dimension
    V: tl.constexpr,  # value dimension
    BTL: tl.constexpr,  # block size along sequence dimension
    BTS: tl.constexpr,  # block size along chunk dimension
    BK: tl.constexpr,   # block size along K dimension
    BV: tl.constexpr,   # block size along V dimension
):
    i_kv = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H

    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    o_k = tl.arange(0, BTS)
    d_h = tl.math.exp2((BTS - o_k) * b_b)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
                            (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
                            (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
                            (0, i_v * BV), (BTS, BV), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)

    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_h[None, :]
        b_o = b_o * tl.math.exp2(b_b * BTS)
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    d_q = tl.math.exp2(tl.arange(0, BTL) * b_b)
    b_o *= d_q[:, None]

    tl.debug_barrier()
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
                            (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
                            (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))

    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(
            m_s,
            tl.math.exp2((o_q[:, None] - o_k[None, :]) * b_b),
            0
        )
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, V), (s_vo_t, s_vo_d),
                            (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def _parallel_retention_bwd_dq(
    i_bh, i_c, i_k, i_v, i_h,
    k, v, do, dq,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
                             (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
                            (0, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t),
                            (i_v * BV, 0), (BV, BTS), (0, 1))

    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_b = tl.math.exp2(b_b * BTS)
    d_h = tl.math.exp2((BTS - tl.arange(0, BTS)) * b_b)

    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * d_h[None, :]
        b_dq *= d_b
        b_dq += tl.dot(b_ds.to(b_v.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))

    b_dq *= tl.math.exp2(tl.arange(0, BTL) * b_b)[:, None] * scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)

    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
                            (i_c * BTL, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t),
                            (i_v * BV, i_c * BTL), (BV, BTS), (0, 1))

    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(
            m_s,
            tl.math.exp2((o_q[:, None] - o_k[None, :]) * b_b),
            0
        )
        b_s = tl.dot(b_do, b_v, allow_tf32=False) * d_s * scale
        b_dq += tl.dot(b_s.to(b_k.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS

    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * s_qk_h, (T, K),
                             (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK),
                             (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def _parallel_retention_bwd_dkv(
    i_bh, i_c, i_k, i_v, i_h,
    q, k, v, do, dk, dv,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_b = tl.math.exp2(b_b * BTS)

    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
                            (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
                            (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dk = tl.zeros([BTL, BK], dtype=tl.float32)
    b_dv = tl.zeros([BTL, BV], dtype=tl.float32)

    d_h = tl.math.exp2((BTL - tl.arange(0, BTL)) * b_b)
    b_kd = (b_k * d_h[:, None]).to(b_k.dtype)
    d_q = tl.math.exp2(tl.arange(0, BTS) * b_b)

    for i in range((tl.cdiv(T, BTS) * BTS) - BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
                                (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t),
                                 (i_v * BV, i), (BV, BTS), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * d_q[None, :]).to(b_do.dtype)

        b_dv *= d_b
        b_s = tl.dot(b_kd.to(b_q.dtype), b_q, allow_tf32=False)
        b_dv += tl.dot(b_s.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)

        b_dk *= d_b
        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        b_dk += tl.dot(b_ds.to(b_q.dtype), tl.trans(b_q), allow_tf32=False)

    b_dk *= d_h[:, None] * scale
    b_dv *= scale

    tl.debug_barrier()
    o_q = tl.arange(0, BTS)
    o_k = tl.arange(0, BTL)

    for i in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
                                (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (
