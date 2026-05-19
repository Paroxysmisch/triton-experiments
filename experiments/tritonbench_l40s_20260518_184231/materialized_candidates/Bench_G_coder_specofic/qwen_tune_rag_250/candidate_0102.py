(K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))
    p_do = tl.make_block_ptr(o + (i_bh+i_k*H*NV+i_v*H) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c*BTL, 0), (BTL, BV), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)

    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BTS, BK] @ [BK, BV] + [BTS, BV]
        b_o = b_o * d_h  # decay the previous acc
        b_s = tl.dot(b_k, b_v, allow_tf32=False)  # [BTS, BTS]
        b_o += tl.dot(b_q, b_s.to(b_q.dtype), allow_tf32=False)  # [BTL, BV]
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        p_do = tl.advance(p_do, (BTS, 0))

    tl.store(p_do, b_o.to(p_do.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _parallel_retention_bwd_dq(
    o_grad,  # output gradient
    k,  # key
    v,  # value
    q,  # query
    do,  # output
    s_qk_h,  # stride size: L * K
    s_qk_t,  # stride size: K
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: L * V
    s_vo_t,  # stride size: V
    s_vo_d,  # stride size: 1
    scale,  # K ** -0.5
    B: tl.constexpr,  # batch_size
    H: tl.constexpr,  # num_heads
    T: tl.constexpr,  # seq_len
    K: tl.constexpr,  # hidden_size
    V: tl.constexpr,  # hidden_size
    BTL: tl.constexpr,  # BLOCK SIZE along the sequence dimension for Q
    BTS: tl.constexpr,  # BLOCK SIZE along the sequence dimension for K/V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
):
    # i_c: chunk index. used for sequence parallelism
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    # decay rate given the head index
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    # cumulative decay from the end of the chunk
    o_k = tl.arange(0, BTS)
    d_h = tl.math.exp2((BTS - o_k) * b_b)

    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_v * BV, 0), (BV, BTS), (0, 1))
    p_o_grad = tl.make_block_ptr(o_grad + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_q *= scale
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)

    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = b_do * d_h
        b_ds = tl.dot(b_do.to(b_k.dtype), b_v, allow_tf32=False)
        b_dq += tl.dot(b_k, b_ds, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        p_do = tl.advance(p_do, (BTS, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))

    b_dq *= scale
    b_dq = (b_dq * d_h).to(b_q.dtype)
    b_oq = tl.dot(b_k, b_v, allow_tf32=False) * scale
    b_dq += tl.dot(o_grad, b_oq, allow_tf32=False) * scale
    tl.store(p_o_grad, b_dq.to(p_o_grad.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _parallel_retention_bwd_dkv(
    o_grad,  # output gradient
    q,  # query
    k,  # key
    v,  # value
    do,  # output
    s_qk_h,  # stride size: L * K
    s_qk_t,  # stride size: K
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: L * V
    s_vo_t,  # stride size: V
    s_vo_d,  # stride size: 1
    scale,  # K ** -0.5
    B: tl.constexpr,  # batch_size
    H: tl.constexpr,  # num_heads
    T: tl.constexpr,  # seq_len
    K: tl.constexpr,  # hidden_size
    V: tl.constexpr,  # hidden_size
    BTL: tl.constexpr,  # BLOCK SIZE along the sequence dimension for Q
    BTS: tl.constexpr,  # BLOCK SIZE along the sequence dimension for K/V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
):
    # i_c: chunk index. used for sequence parallelism
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    # decay rate given the head index
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    # cumulative decay from the end of the chunk
    o_k = tl.arange(0, BTS)
    d_h = tl.math.exp2((BTS - o_k) * b_b)

    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_o_grad = tl.make_block_ptr(o_grad + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t
