import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd
from fla.utils import contiguous

@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    q, k, v, w, u, o, initial_state, final_state,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    T, scale,
    BK: tl.constexpr, BV: tl.constexpr, NK: tl.constexpr, NV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr, REVERSE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * s_qk_h + i_k * NK + tl.arange(0, BK) + ((T-1) if REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * NK + tl.arange(0, BK) + ((T-1) if REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * NV + tl.arange(0, BV) + ((T-1) if REVERSE else 0)
    p_o = o + (i_bh + i_k * BV * i_bh) * s_vo_h + i_v * NV + tl.arange(0, BV) + ((T-1) if REVERSE else 0)

    b_k = tl.load(p_k, mask=(tl.arange(0, BK) + ((T-1) if REVERSE else 0) < T) & (i_k * NK + tl.arange(0, BK) < k.size(1)), other=0.0).to(tl.float32)
    b_v = tl.load(p_v, mask=(tl.arange(0, BV) + ((T-1) if REVERSE else 0) < T) & (i_v * NV + tl.arange(0, BV) < v.size(1)), other=0.0).to(tl.float32)
    b_u = tl.load(u + i_bh * T + tl.arange(0, BK) + ((T-1) if REVERSE else 0), mask=tl.arange(0, BK) + ((T-1) if REVERSE else 0) < T, other=0.0).to(tl.float32)
    b_h = tl.zeros([BV], dtype=tl.float32)
    b_o = tl.zeros([BV], dtype=tl.float32)

    for _ in range(0, T):
        _k = tl.load(p_k, mask=(tl.arange(0, BK) < T) & (i_k * NK + tl.arange(0, BK) < k.size(1)), other=0.0).to(tl.float32)
        _v = tl.load(p_v, mask=(tl.arange(0, BV) < T) & (i_v * NV + tl.arange(0, BV) < v.size(1)), other=0.0).to(tl.float32)
        _q = tl.load(p_q, mask=(tl.arange(0, BK) < T) & (i_k * NK + tl.arange(0, BK) < k.size(1)), other=0.0).to(tl.float32) * scale
        _u = tl.load(u + i_bh * T + tl.arange(0, BK), mask=tl.arange(0, BK) < T, other=0.0).to(tl.float32)

        b_k += _k
        b_v += _v
        b_u += _u
        b_h = b_h * tl.exp(b_u) + b_v * tl.exp(b_k)
        b_o = b_o * tl.exp(b_u) + b_v * tl.exp(b_k - b_u) * _q
        b_o = b_o / b_h

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=(tl.arange(0, BV) < T) & (i_v * NV + tl.arange(0, BV) < v.size(1)))
        b_o += tl.zeros([BV], dtype=tl.float32)

        p_q += -BK if REVERSE else BK
        p_k += -BK if REVERSE else BK
        p_v += -BV if REVERSE else BV
        p_o += -BV if REVERSE else BV

    if USE_INITIAL_STATE:
        b_h += tl.exp(tl.load(initial_state + i_bh * 2 * NK + tl.arange(0, BK), mask=tl.arange(0, BK) < NK, other=0.0).to(tl.float32) + tl.load(initial_state + i_bh * 2 * NK + NK + tl.arange(0, BK), mask=tl.arange(0, BK) < NK, other=0.0).to(tl.float32))
        b_o += tl.exp(tl.load(initial_state + i_bh * 2 * NK + NK + tl.arange(0, BK), mask=tl.arange(0, BK) < NK, other=0.0).to(tl.float32)) * tl.load(p_q, mask=(tl.arange(0, BK) < T) & (i_k * NK + tl.arange(0, BK) < k.size(1)), other=0.0).to(tl.float32) * scale
        b_o = b_o / b_h

        tl.store(o + i_bh * s_vo_h + i_v * NV + tl.arange(0, BV) + ((T-1) if REVERSE else 0), b_o.to(p_o.dtype.element_ty), mask=(tl.arange(0, BV) < T) & (i_v * NV + tl.arange(0, BV) < v.size(1)))

    if STORE_FINAL_STATE:
        tl.store(final_state + i_bh * 2 * NK + tl.arange(0, BK) + NK, b_h.to(final_state.dtype.element_ty), mask=tl.arange(0, BK) + NK < 2 * NK)
        tl.store(final_state + i_bh * 2 * NK + tl.arange(0, BK), b_o.to(final_state.dtype.element_ty) * tl.exp(tl.load(u + i_bh * T + tl.arange(0, BK) + ((T-1) if REVERSE else 0), mask=tl.arange(0, BK) + ((T-1) if REVERSE else 0) < T, other=0.0).to(tl.float32)), mask=tl.arange(0, BK) < NK)

class FusedRecurrentRWKV6Function(torch.autograd.Function):
    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, q, k, v, w, u, scale, initial_state, output_final_state):
        B, H, T, K = q.shape
        _, _, _, V = v.shape
        NK, BV = 32, 32

        BK = min(triton.next_power_of_2(K), NK)
        NV = min(triton.next_power_of_2(V), BV)

        grid = (NV, BK, B * H)
        p_o = torch.empty(B, H, T, V, dtype=q.dtype, device=q.device)
        initial_state = initial_state if initial_state is not None else torch.empty(B * H, 2 * K, dtype=torch.float32, device=q.device)
        final_state = torch.empty(B * H, 2 * K, dtype=torch.float32, device=q.device) if output_final_state else None

        BK, BV = NK, NV
        fused_recurrent_rwkv6_fwd_kernel[grid](
            q, k, v, w, u, p_o, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            T, scale,
            BK=BK, BV=BV, NK=NK, NV=NV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            REVERSE=False,
        )

        ctx.save_for_backward(q, k, v, w, u, initial_state)
        ctx.scale = scale

        return p_o if not output_final_state else (p_o, final_state)

def fused_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=None, output_final_state=False):
    return FusedRecurrentRWKV6Function.apply(q, k, v, w, u, scale, initial_state, output_final_state)
