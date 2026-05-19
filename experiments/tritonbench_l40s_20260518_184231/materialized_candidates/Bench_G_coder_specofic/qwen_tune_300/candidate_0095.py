import torch
import triton
import triton.language as tl
from packaging import version

@triton.jit
def fused_recurrent_retention_fwd_kernel(
    q, k, v, o, initial_state, final_state,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)

    if USE_INITIAL_STATE:
        p_h = initial_state + i_bh * H * BK + i_k * BK + tl.arange(0, BK)
        h = tl.load(p_h).to(tl.float32)
    else:
        h = tl.zeros([BK], dtype=tl.float32)

    mask = (i_k * BK + tl.arange(0, BK)) < tl.cdiv(q.shape[1], BK)
    for i in range(0, tl.cdiv(T, 1)):
        _mask = mask & ((i * 1 + i_k * BK + tl.arange(0, BK)) < tl.cdiv(q.shape[1], BK))
        _p_q = p_q + i * 1 * s_qk_t
        _p_k = p_k + i * 1 * s_qk_t
        _p_v = p_v + i * 1 * s_vo_t
        _p_o = p_o + i * 1 * s_vo_t

        q = tl.load(_p_q, mask=_mask, other=0).to(tl.float32)
        k = tl.load(_p_k, mask=_mask, other=0).to(tl.float32)
        v = tl.load(_p_v, mask=_mask, other=0).to(tl.float32)
        h = h * tl.sigmoid(q) + k
        o = tl.dot(h.to(p_o.dtype.element_ty)[:, None], v[None, :], allow_tf32=False)
        tl.store(_p_o, o.to(p_o.dtype.element_ty), mask=_mask)

    if STORE_FINAL_STATE:
        p_final = final_state + i_bh * H * BK + i_k * BK + tl.arange(0, BK)
        tl.store(p_final, h.to(p_final.dtype.element_ty))


@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q, k, v, do, dq, dk, dv, initial_state,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)

    if USE_INITIAL_STATE:
        p_h = initial_state + i_bh * H * BK + i_k * BK + tl.arange(0, BK)
        h = tl.load(p_h).to(tl.float32)
    else:
        h = tl.zeros([BK], dtype=tl.float32)

    mask = (i_k * BK + tl.arange(0, BK)) < tl.cdiv(q.shape[1], BK)
    for i in range(0, tl.cdiv(T, 1)):
        _mask = mask & ((i * 1 + i_k * BK + tl.arange(0, BK)) < tl.cdiv(q.shape[1], BK))
        _p_do = p_do + i * 1 * s_vo_t
        _p_dq = p_dq + i * 1 * s_qk_t
        _p_dk = p_dk + i * 1 * s_qk_t
        _p_v = p_v + i * 1 * s_vo_t
        _p_dv = p_dv + i * 1 * s_vo_t

        do = tl.load(_p_do, mask=_mask, other=0).to(tl.float32)
        k = tl.load(_p_k, mask=_mask, other=0).to(tl.float32)
        v = tl.load(_p_v, mask=_mask, other=0).to(tl.float32)
        o = tl.dot(h.to(p_do.dtype.element_ty)[:, None], v[None, :], allow_tf32=False)
        dq = tl.dot(k[:, None], do[None, :], allow_tf32=False)
        h = h * tl.sigmoid(q) + k
        dv = tl.dot(h.to(p_do.dtype.element_ty)[:, None], do[None, :], allow_tf32=False)
        dq *= scale * tl.sigmoid(o) * (1 - tl.sigmoid(o))
        tl.store(_p_dq, dq.to(p_dq.dtype.element_ty), mask=_mask)
        tl.store(_p_dk, dq.to(p_dk.dtype.element_ty), mask=_mask)
        tl.store(_p_dv, dv.to(p_dv.dtype.element_ty), mask=_mask)

class FusedRecurrentRetentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, initial_state=None, output_final_state=False):
        ctx.save_for_backward(q, k, v, initial_state)

        B, H, T, D = q.shape
        scale = D ** -0.5
        BK = 4
        BV = 8

        def num_warps(H):
            if H <= 16:
                return 4
            elif H <= 32:
                return 8
            else:
                return 16

        o = torch.empty(B, H, T, D, device=q.device, dtype=q.dtype)
        final_state = torch.empty(B, H, D, device=q.device, dtype=torch.float32) if output_final_state else None

        grid = (triton.cdiv(D, BV), triton.cdiv(D, BK), B * H)
        fused_recurrent_retention_fwd_kernel[grid](
            q, k, v, o, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            B, H, T, scale,
            BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            num_warps=num_warps,
        )
        return o, final_state if output_final_state else o

    @staticmethod
    def backward(ctx, do, d_final_state=None):
        q, k, v, initial_state = ctx.saved_tensors

        B, H, T, D = q.shape
        scale = D ** -0.5
        BK = 4
        BV = 8

        def num_warps(H):
            if H <= 16:
                return 4
            elif H <= 32:
                return 8
            else:
                return 16

        dq = torch.empty_like(q)
        dk = torch.empty_like(q)
        dv = torch.empty_like(v)
        grid = (triton.cdiv(D, BV), triton.cdiv(D, BK), B * H)
        fused_recurrent_retention_bwd_kernel[grid](
            q, k, v, do, dq, dk, dv, initial_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            B, H, T, scale,
            BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            num_warps=num_warps,
        )
        return dq, dk,
