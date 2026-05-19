scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BD, BS] block D, [BS, BV] block S
        # [BD, BV] = [BD, BS] @ [BS, BV]
        b_s = tl.dot(b_q.to(b_k.dtype), b_k.to(b_k.dtype), allow_tf32=False) * d_h[None, :]
        b_o = b_o * (1 - d_h[None, :])
        # [BD, BV] += [BS, BV]
        b_o += tl.dot(b_s.to(b_v.dtype), b_v.to(b_v.dtype), allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    b_o = b_o.to(o.dtype.element_ty)
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def parallel_retention_bwd_kernel(
    q,  # [B, H, L, K]
    k,  # [B, H, L, V]
    v,  # [B, H, L, V]
    do,  # [B, H, L, V]
    dq,  # [B, H, L, K]
    dk,  # [B, H, L, K]
    dv,  # [B, H, L, V]
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
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
    # i_c: chunk index. used for sequence parallelism
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    # decay rate given the head index
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    o_k = tl.arange(0, BTS)
    d_h = tl.math.exp2((BTS - o_k) * b_b)

    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    # [BQ, BD] block Q, in the shared memory throughout the whole kernel
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_bq = b_q * scale
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    for _ in range(0, i_c * BTL, BTS):
        # [BD, BS] block D, [BS, BK] block K
        # [BD, BK] = [BD, BS] @ [BS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BD, BV] = [BD, BV] + ([BD, BS] @ [BS, BV])
        b_ds = tl.dot(b_bq.to(b_k.dtype), b_k.to(b_k.dtype), allow_tf32=False) * d_h[None, :]
        b_dq = b_dq * (1 - d_h[None, :])
        b_dq += tl.dot(b_ds.to(b_v.dtype), b_v.to(b_v.dtype), allow_tf32=False)
        b_dq += tl.dot(b_do.to(b_k.dtype), b_k.to(b_k.dtype), allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        p_do = tl.advance(p_do, (BTS, 0))
    b_dq = b_dq.to(dq.dtype.element_ty)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    # [BS, BK] block S, [BS, BV] block V
    # [BS, BV] = [BS, BK] @ [BK, BV]
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (K, T), (s_vo_d, s_vo_t), (i_k * BK, 0), (BV, BTS), (0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dk = tl.zeros([BTS, BK], dtype=tl.float32)
    b_dv = tl.zeros([BV, BTS], dtype=tl.float32)
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_k * BK), (BTS, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (K, T), (s_vo_d, s_vo_t), (i_k * BK, 0), (BV, BTS), (0, 1))
    for _ in range(0, i_c * BTL, BTS):
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS, BV] = [BS, BV] + ([BS, BK] @ [BK, BV])
        b_dk += tl.dot(b_do.to(b_v.dtype), b_v.to(b_v.dtype), allow_tf32=False)
        b_dv += tl.dot(b_do.to(b_k.dtype), b_k.to(b_k.dtype), allow_tf32=False)
        p_do = tl.advance(p_do, (BTS, 0))
    b_dk = (b_dk * d_h[None, :]).to(dk.dtype.element_ty)
    b_dv = (b_dv * d_h[:, None]).to(dv.dtype.element_ty)
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))


class ParallelRetentionFunction(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, q, k, v, scale=1):
        BTL, BTS = 128, 64
        assert BTL % BTS == 0
        BK = min(128, triton.next_power_of_2(k.shape[-1]))
        BV = min(128, triton.next_power_of_2(v.shape[-1]))
        BK, BV = max(BK, 16), max(BV, 16)
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head
