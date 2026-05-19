import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def fused_recurrent_retention_fwd_kernel(
    q, k, v, o, initial_state, final_state,
    s_qk_h, s_qk_t, s_qk_d,  # Strides for Q and K
    s_v_h, s_v_t, s_v_d,      # Strides for V
    s_o_h, s_o_t, s_o_d,      # Strides for O
    B, H, T, scale,
    BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = 1.0 - tl.math.pow(2.0, -5.0 - i_h * 1.0)  # Decay factor per head

    # Compute pointers for Q, K, V blocks
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV)
    p_o = o + i_bh * s_o_h + i_k * (B * H * s_o_h) + i_v * BV + tl.arange(0, BV)

    # Masks to handle boundary conditions
    mask_bk = (i_k * BK + tl.arange(0, BK)) < DK
    mask_bv = (i_v * BV + tl.arange(0, BV)) < DV
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    h = tl.zeros([BV, BK], dtype=tl.float32)  # Accumulator

    # Load initial state if provided
    if USE_INITIAL_STATE:
        p_init = initial_state + i_bh * DK * DV + (i_k * BK + tl.arange(0, BK)[None, :]) * DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init, mask=mask_kv, other=0).to(tl.float32)

    # Iterate over sequence
    for t in range(T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale

        h = b_b * h + _k[None, :] * _v[:, None]  # Update accumulator
        _o = tl.sum(h * _q[None, :], axis=1)      # Compute output
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)

        # Move pointers by respective time strides
        p_q += s_qk_t
        p_k += s_qk_t
        p_v += s_v_t
        p_o += s_o_t

    # Store final state if needed
    if STORE_FINAL_STATE:
        p_final = final_state + i_bh * DK * DV + (i_k * BK + tl.arange(0, BK)[None, :]) * DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final, h.to(p_final.dtype.element_ty), mask=mask_kv)

@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q, k, v, do, dq, dk, dv, initial_state,
    s_qk_h, s_qk_t, s_qk_d,  # Q/K strides
    s_v_h, s_v_t, s_v_d,      # V strides
    s_do_h, s_do_t, s_do_d,   # dO strides
    B, H, T, scale,
    BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = 1.0 - tl.math.pow(2.0, -5.0 - i_h * 1.0)

    # Initialize pointers
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * s_do_h + i_v * BV + tl.arange(0, BV)
    p_dq = dq + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < DK
    mask_bv = (i_v * BV + tl.arange(0, BV)) < DV

    h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_init = initial_state + i_bh * DK * DV + (i_k * BK + tl.arange(0, BK)[:, None]) * DV + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_init, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)

    # Forward pass to compute dQ
    for t in range(T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)

        h = b_b * h + _k[:, None] * _v[None, :]
        d_q = tl.sum(h * _do[None, :], axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_k += s_qk_t
        p_do += s_do_t
        p_v += s_v_t
        p_dq += s_qk_t

    tl.debug_barrier()

    # Reverse pass to compute dK and dV
    h = tl.zeros([BK, BV], dtype=tl.float32)
    for t in reversed(range(T)):
        p_q_t = p_q + t * s_qk_t
        p_k_t = p_k + t * s_qk_t
        p_v_t = p_v + t * s_v_t
        p_do_t = p_do + t * s_do_t

        _q = tl.load(p_q_t, mask=mask_bk, other=0).to(tl.float32) * scale
        _do = tl.load(p_do_t, mask=mask_bv, other=0).to(tl.float32)
        _k = tl.load(p_k_t, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v_t, mask=mask_bv, other=0).to(tl.float32)

        h = h * b_b + _q[:, None] * _do[None, :]
        d_k = tl.sum(h * _v[None, :], axis=1)
        d_v = tl.sum(h * _k[:, None], axis=0)

        p_dk_t = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + t * s_qk_t
        p_dv_t = dv + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + t * s_v_t
        tl.store(p_dk_t, d_k.to(p_dk_t.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv_t, d_v.to(p_dv_t.dtype.element_ty), mask=mask_bv)

class FusedRecurrentRetention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state=None, output_final_state=False):
        B, H, T, DK = q.shape
        DV = v.shape[-1]
        scale = DK ** -0.5

        BK = min(DK, 32)
        BV = min(DV, 32)
        NK, NV = triton.cdiv(DK, BK), triton.cdiv(DV, BV)

        o = torch.empty((NK, B, H, T, DV), device=q.device, dtype=q.dtype)
        final_state = torch.empty((B, H, DK, DV), device=q.device, dtype=q.dtype) if output_final_state else None

        grid = (NV, NK, B * H)
        fused_recurrent_retention_fwd_kernel[grid](
            q, k, v, o, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),  # Q/K strides
            v.stride(1), v.stride(2), v.stride(3),  # V strides
            o.stride(2), o.stride(3), o.stride(4),  # O strides
            B, H, T, scale, BK=BK, BV=BV, DK=DK, DV=DV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            num_warps=1,
            num_stages=1
        )

        ctx.save_for_backward(q, k, v, initial_state)
        ctx.scale = scale
        return o.sum(0), final_state

    @staticmethod
    def backward(ctx, do, d_final_state=None):
        q, k, v, initial_state = ctx.saved_tensors
        B, H, T, DK = q.shape
        DV = v.shape[-1]
        scale = ctx.scale

        BK = min(DK, 32)
        BV = min(DV, 32)
        NV, NK = triton.cdiv(DV, BV), triton.cdiv(DK, BK)

        dq = torch.empty((NV, B, H, T, DK), device=q.device, dtype=q.dtype)
        dk = torch.empty_like(q)
        dv = torch.empty((NK, B, H, T, DV), device=v.device, dtype=v.dtype)

        grid = (NV, NK, B * H)
        fused_recurrent_retention_bwd_kernel[grid](
            q, k, v, do, dq, dk, dv, initial_state,
            q.stride(1), q.stride(2), q.stride(3),  # Q/K strides
            v.stride(1), v.stride(2), v.stride(3),  # V strides
            do.stride(1), do.stride(2), do.stride(3),  # dO strides
            B, H, T, scale, BK=BK, BV=BV, DK=DK, DV=DV,
            USE_INITIAL_STATE=initial_state is not None,
            num_warps=1,
            num_stages=1
        )

        return dq.sum(0), dk.sum(0), dv.sum(0), None, None

def fused_recurrent_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    return FusedRecurrentRetention.apply(q, k, v, initial_state, output_final_state)
