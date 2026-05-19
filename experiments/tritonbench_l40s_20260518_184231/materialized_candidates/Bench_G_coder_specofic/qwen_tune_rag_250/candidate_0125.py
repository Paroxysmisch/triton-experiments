.arange(0, BTS)
    o_k_mask = o_k < (T - i_c * BTL)
    b_d = tl.math.exp2(o_k * b_b)
    b_d = tl.where(o_k_mask, b_d, 0)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)

    for _ in range(0, i_v + 1):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # d(k . q)
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s = b_s * b_d[None, :] + tl.where((i_c == 0) & (o_k_mask & (o_k == 0)), 1, 0)
        # o = softmax(s) * v
        b_o = tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False) + b_o
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    tl.debug_barrier()
    b_o = b_o.to(o.dtype.element_ty)
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _parallel_retention_bwd_dq(
    i_bh, i_c, i_k, i_v, i_h,
    q, k, v, do, dz, o,  # same as fwd
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    scale,
    BTL, BTS, BK, BV,  # BLOCK SIZE along the sequence dimension for Q
    BT: tl.constexpr,
):
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    # cumulative decay from the end of the chunk
    o_k = tl.arange(0, BT)
    o_k_mask = o_k < (BTL - i_c * BTL)
    b_d = tl.math.exp2(o_k * b_b)
    b_d = tl.where(o_k_mask, b_d, 0)

    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_dz = tl.zeros([BTL], dtype=tl.float32)
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)

    p_kv = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))

    for _ in range(0, i_v + 1):
        b_k = tl.load(p_kv, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # d(s) * v
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s = (b_s * b_d[:, None] + tl.where((i_c == 0) & (o_k_mask & (o_k == 0)), b_d[None, :], 0)).to(b_ds.dtype)
        b_dz += tl.sum(b_ds * b_s, axis=1)
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)
        p_kv = tl.advance(p_kv, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
    b_dq = b_dq.to(q.dtype.element_ty)
    tl.debug_barrier()
    p_dq = tl.make_block_ptr(dz + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    p_dq = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_dq, boundary_check=(0, 1))
    b_dq = b_dq + (b_q * b_dz[:, None]).to(b_q.dtype)
    b_dq = b_dq.to(dq.dtype.element_ty)
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))


@triton.jit
def _parallel_retention_bwd_dkv(
    i_bh, i_c, i_k, i_v, i_h,
    q, k, v, do, dz, o,  # same as fwd
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    scale,
    BTL, BTS, BK, BV,  # BLOCK SIZE along the sequence dimension for Q
    BT: tl.constexpr,
):
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    # cumulative decay from the end of the chunk
    o_k = tl.arange(0, BT)
    o_k_mask = o_k < (BTL - i_c * BTL)
    b_d = tl.math.exp2(o_k * b_b)
    b_d = tl.where(o_k_mask, b_d, 0)

    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_do = tl.load(p_do
