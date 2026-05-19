import torch
import triton
import triton.language as tl

@triton.jit
def parallel_rebased_fwd_kernel(
    q, k, v, o, z, 
    s_qk_h, s_qk_t, s_qk_d, 
    s_vo_h, s_vo_t, s_vo_d, 
    B, H, T, scale, 
    BTL: tl.constexpr, BTS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, 
    DK: tl.constexpr, DV: tl.constexpr
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), 
                            (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), 
                            (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), 
                            (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)

    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = b_s * b_s
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

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
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, DV), 
                            (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c * BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_z, b_z.to(p_z.dtype.element_ty), 
             mask=((i_c * BTL + tl.arange(0, BTL)) < T))


@triton.jit
def parallel_rebased_bwd_kernel(
    q, k, v, do, dz, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, 
    s_vo_t, s_vo_d, B, H, T, scale, 
    BTL: tl.constexpr, BTS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, 
    DK: tl.constexpr, DV: tl.constexpr
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H
    _parallel_rebased_bwd_dq(
        i_bh, i_c, i_k, i_v, i_h, 
        q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, 
        s_vo_t, s_vo_d, B, H, T, scale, BTL=BTL, BTS=BTS, BK=BK, BV=BV, DK=DK, DV=DV
    )
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(
        i_bh, i_c, i_k, i_v, i_h, 
        q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, 
        s_vo_t, s_vo_d, B, H, T, scale, BTL, BTS, BK, BV, DK, DV
    )

class ParallelBasedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        BTL, BTS = 128, 32
        assert BTL % BTS == 0
        assert q.dtype == v.dtype
        BK = min(128, triton.next_power_of_2(k.shape[-1]))
        BV = min(128, triton.next_power_of_2(v.shape[-1]))
        BK, BV = max(BK, 16), max(BV, 16)
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        num_stages = 2
        num_warps = 4
        NK = triton.cdiv(d_head_qk, BK)
        NV = triton.cdiv(d_head_v, BV)
        grid = (NK * NV, triton.cdiv(seq_len, BTL), batch_size * n_heads)

        assert NK == 1, "will encounter some synchronization issue if not."

        o = torch.empty(NK, batch_size, n_heads, seq_len, 
                        d_head_v, device=q.device)
        z = torch.empty(NK, batch_size, n_heads, seq_len, 
                        device=q.device)
        parallel_rebased_fwd_kernel[grid](
            q, k, v, o, z, 
            q.stride(1), q.stride(2), q.stride(3), 
            v.stride(1), v.stride(2), v.stride(3), 
            batch_size, n_heads, seq_len, scale, 
            BTL=BTL, BTS=BTS, BK=BK, BV=BV, DK=d_head_qk, DV=d_head_v, 
            num_warps=num_warps, 
            num_stages=num_stages
        )
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        return o.sum(0).to(q.dtype), z.sum(0).to(q.dtype)

    @staticmethod
    def backward(ctx, do, dz):
        q, k, v = ctx.saved_tensors
        scale = ctx.scale
        BTL, BTS = 64, 32
        assert BTL % BTS == 0
        BK = min
