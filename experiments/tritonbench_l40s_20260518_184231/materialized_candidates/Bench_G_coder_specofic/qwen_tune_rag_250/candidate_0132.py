BTL, i_v * BV), (BTS, BV), (1, 0))

    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)

        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, DV), 
                            (s_vo_t, s_vo_d), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c * BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_z, b_z.to(p_z.dtype.element_ty), mask=((i_c * BTL + tl.arange(0, BTL)) < T)))


@triton.jit
def parallel_rebased_bwd_kernel(
    q, k, v, do, dz, dq, dk, dv, 
    s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, 
    B, H, T, scale, 
    BTL: tl.constexpr, BTS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, 
    DK: tl.constexpr, DV: tl.constexpr
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H

    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), 
                             (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + (i_bh) * s_qk_h, (T, DK), 
                            (s_qk_t, s_qk_d), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_k) * s_qk_h, (T, DK), 
                             (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    b_k = tl.load(k + i_bh * s_qk_h + i_k * BK * s_qk_d, boundary_check=(0, 1)).to(b_q.dtype)
    b_v = tl.load(v + i_bh * s_vo_h + i_v * BV * s_vo_d, boundary_check=(0, 1)).to(b_q.dtype)
    b_dz = tl.load(dz + i_bh * T + i_c * BTL + tl.arange(0, BTL))
    b_s = tl.dot(b_do, b_v, allow_tf32=False)
    if i_h == i_k:
        b_s = b_s + tl.dot(b_do, b_v, allow_tf32=False) * b_q * scale
    b_ds = b_dz[:, None] * b_s * b_s
    b_dk = tl.sum(b_ds.to(b_k.dtype), axis=1)
    b_dv = tl.sum(b_ds.to(b_v.dtype), axis=0)
    b_dq += b_ds.to(b_q.dtype)
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, DK), 
                             (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTS, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, DV), 
                             (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        p_dk = tl.advance(p_dk, (0, BTS))
        p_dv = tl.advance(p_dv, (BTS, 0))
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), 
                            (s_qk_d, s_qk_t), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), 
                            (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = b_s * b_s
        b_s = b_s * b_do
        if i_h == i_k:
            b_s = b_s + tl.dot(b_do, b_v, allow_tf32=False) * m_s * b_q * scale
        b_ds = b_dz[:, None] * b_s
        b_dk = tl.sum(b_ds.to(b_k.dtype), axis=1)
        b_dv = tl.sum(b_ds.to(b_v.dtype), axis=0)
        b_dq += tl.dot(b_ds.to(b_q.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    b_dk += tl.sum(b_dq.to(b_k.dtype), axis=1)
    b_dv += tl.sum(b_dq.to(b_v.dtype), axis=0)
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (DK, T), 
                             (s_qk_d, s_qk_t), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, DV), 
                             (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class ParallelBasedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False):
        BTL, BTS = 128, 32
        BK = min(128, triton.next_power_of_2(k.shape[-1]))
        BV = min(128, triton.next_power_of_2(v.shape[-1]))
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        num_stages = 2
        num_warps = 4
        NK = triton.cdiv(d_head_qk, BK)
        NV =
