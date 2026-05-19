import torch
import triton
import triton.language as tl

@triton.jit
def parallel_retention_fwd_kernel(
    q, k, v, o, initial_state, final_state,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh+i_k*B*H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK, (DK,), (DK,), (i_k * BK,), (BK,), (1,))
        h = tl.load(p_h, boundary_check=(0,), padding_option='zero')
    
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        b_q = tl.load(p_q, boundary_check=(0,), padding_option='zero')
        b_k = tl.load(p_k, boundary_check=(0,), padding_option='zero')
        b_v = tl.load(p_v, boundary_check=(0,), padding_option='zero')
        
        if i == 0 and USE_INITIAL_STATE:
            b_q = b_q.to(tl.float32)
            b_k = b_k.to(tl.float32)
            b_v = b_v.to(tl.float32)
            b_h = h[None, :] * b_q[:, None]
            b_o = b_o * 0 + b_v * tl.math.exp2(16) + tl.sum(b_k * b_h, axis=1)[:, None]
        else:
            b_o = b_o * tl.math.exp2(16) + tl.sum(b_k * (b_q * b_o.to(b_q.dtype) / scale), axis=1)[:, None] + b_v
        
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))
        
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
    
    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(final_state + i_bh * DK, (DK,), (DK,), (i_k * BK,), (BK,), (1,))
        tl.store(p_final, (b_o * tl.math.exp2(-16)).to(p_final.dtype.element_ty), boundary_check=(0,))

class ParallelRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, initial_state, output_final_state):
        B, H, T, DK, DV = *q.shape, v.shape[-1]
        BT = 32
        BK, BV = min(DK, 64), min(DV, 64)
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(DK, BK), triton.cdiv(DV, BV)
        num_stages = 1
        num_warps = 4 if BK == 64 else 2

        o = torch.empty(NK, B, H, T, DV, device=q.device, dtype=q.dtype)
        if output_final_state:
            final_state = torch.empty(B, H, DK, device=q.device, dtype=torch.float32)
        else:
            final_state = None
        
        grid = (NV, NK, NT)
        parallel_retention_fwd_kernel[grid](
            q, k, v, o, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            B, H, T, scale,
            BT, BK, BV, DK, DV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            num_warps=num_warps,
            num_stages=num_stages
        )
        
        ctx.save_for_backward(q, k, v, initial_state)
        return o.sum(0).to(q.dtype), final_state

    @staticmethod
    def backward(ctx, do, d_final_state=None):
        q, k, v, initial_state = ctx.saved_tensors
        B, H, T, DK, DV = *q.shape, v.shape[-1]
        BT = 32
        BK, BV = min(DK, 64), min(DV, 64)
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(DK, BK), triton.cdiv(DV, BV)
        num_stages = 1
        num_warps = 4 if BK == 64 else 2

        dq = torch.empty(NV, B, H, T, DK, device=q.device, dtype=torch.float32)
        dk = torch.empty(NV, B, H, T, DK, device=q.device, dtype=torch.float32)
        dv = torch.empty(NV, B, H, T, DV, device=q.device, dtype=v.dtype)
        grid = (NV, NK, NT)

        parallel_retention_bwd_kernel[grid](
            q, k, v, do, initial_state, dq, dk, dv,
            s_qk_h=q.stride(1), s_qk_t=q.stride(2), s_qk_d=q.stride(3),
            s_vo_h=v.stride(1), s_vo_t=v.stride(2), s_vo_d=v.stride(3),
            B=B, H=H, T=T, scale=1,
            BT=BT, BK=BK, BV=BV, DK=DK, DV=DV,
            USE_INITIAL_STATE=initial_state is not None,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq.sum(0).to(q.dtype), dk.sum(0).to(k.dtype), dv.sum(0).to(v.dtype), None, None, None

@triton.jit
def _parallel_retention_bwd_dq(
    i_bh, i_k, i_v, i_h,
    q, k, v, do, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr
):
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_dq = tl.make_block_ptr(dq + (i_bh + i_k * B * H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0,), padding_option='zero')
    b_do = tl.load(p_do, boundary_check=(0,), padding_option='zero')
    b_dq = tl.sum(b_do * b_q.to(b_do.dtype) / scale, axis=1)[:, None]
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0,))

@triton.jit
def _parallel_retention_bwd_dkv(
    i_bh, i_k, i_v, i_h,
    q, k, v, do, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d,
    B, H, T
