import torch
import triton
import triton.language as tl

@triton.jit
def attention_fwd_kernel(
    Q, K, V, sm_scale,
    B_H,
    BT,
    mid_dim,
    H,
    T,
    O,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    s_b_h,
    BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    IFCOND: tl.constexpr, STORE: tl.constexpr,
    BT_H: tl.constexpr, BT_T: tl.constexpr,
    BK_H: tl.constexpr, BK_T: tl.constexpr,
    BV_H: tl.constexpr, BV_T: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BT_H, BT_T, BK_H, BV_H], dtype=tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(Q + i_bh * s_qk_h, (mid_dim, T), (s_qk_d, s_qk_t), (i_k * BK, i * BT), (BK_H, BT_T), (0, 1))
        p_k = tl.make_block_ptr(K + i_bh * s_qk_h, (T, mid_dim), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT_T, BK_H), (1, 0))
        p_v = tl.make_block_ptr(V + i_bh * s_vo_h, (mid_dim, DV), (s_vo_d, s_vo_t), (i_k * BK, i_v * BV), (BK_H, BV_H), (0, 1))
        p_h = tl.make_block_ptr(B_H + i_bh * s_b_h, (mid_dim, T), (s_b_h, 0), (i_k * BK, i * BT), (BK_H, BT_T), (0, 1))
        p_o = tl.make_block_ptr(O + (i_bh + i_k * B * H) * s_vo_h, (mid_dim, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV_H, BT_T), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = (b_q * sm_scale).to(b_k.dtype)
        b_s = tl.zeros([BT_H, BT_T, BV_H], dtype=tl.float32)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
        b_o = tl.zeros([BT_H, BT_T, BV_H], dtype=tl.float32)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        if IFCOND:
            b_h = tl.where(i_k == 0, b_o, b_h + b_o)
        else:
            b_h += b_o
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        if STORE:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


class AttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, h, at, b, hd, nheads, mid_dim, d, is_training, cond=None):
        scale = math.sqrt(hd)
        sm_scale = scale
        BT, BK, BV = triton.next_power_of_2(at), triton.next_power_of_2(d // nheads), min(triton.next_power_of_2(d // nheads), 64)
        NT = triton.cdiv(at, BT)
        NK = triton.cdiv(d, BK)
        N = triton.cdiv(d, BV)
        BK, BV = min(BK, 64), min(BV, 64)
        num_stages = 4 if BT <= 16 else 3
        num_warps = 4
        BK_H = min(BK, 32)
        BV_H = min(BV, 32)
        BT_H = min(BT, 32)
        o = torch.empty((b, hd, mid_dim, at), device=q.device, dtype=q.dtype)
        grid = (NK, nheads, b)
        if cond is None:
            attention_fwd_kernel[grid](
                q, k, v, sm_scale,
                h,
                BT,
                mid_dim, hd, at, o,
                q.stride(0), q.stride(2), q.stride(3),
                v.stride(0), v.stride(2), v.stride(3),
                h.stride(0),
                BK=BK, BV=BV,
                DK=k.shape[-1], DV=v.shape[-1],
                IFCOND=False, STORE=is_training,
                BT_H=BT_H, BK_H=BK_H, BV_H=BV_H,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            attention_fwd_kernel[grid](
                q, k, v, sm_scale,
                h,
                BT,
                mid_dim, hd, at, o,
                q.stride(0), q.stride(2), q.stride(3),
                v.stride(0), v.stride(2), v.stride(3),
                h.stride(0),
                BK=BK, BV=BV,
                DK=k.shape[-1], DV=v.shape[-1],
                IFCOND=True, STORE=is_training,
                BT_H=BT_H, BK_H=BK_H, BV_H=BV_H,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        ctx.save_for_backward(q, k, v, h, cond)
        ctx.scale = scale
        ctx.sm_scale = sm_scale
        return o
