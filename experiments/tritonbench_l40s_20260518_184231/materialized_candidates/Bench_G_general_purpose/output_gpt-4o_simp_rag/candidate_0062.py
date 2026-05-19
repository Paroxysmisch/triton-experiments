import torch
import triton
import triton.language as tl

@triton.jit
def fused_recurrent_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    scale, B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # Compute pointers
    p_q = q_ptr + i_bh * H * K + i_k * BK + tl.arange(0, BK)
    p_k = k_ptr + i_bh * H * K + i_k * BK + tl.arange(0, BK)
    p_v = v_ptr + i_bh * H * V + i_v * BV + tl.arange(0, BV)
    p_o = o_ptr + i_bh * H * V + i_v * BV + tl.arange(0, BV)

    # Load data
    q = tl.load(p_q, mask=(i_k * BK + tl.arange(0, BK)) < K, other=0.0)
    k = tl.load(p_k, mask=(i_k * BK + tl.arange(0, BK)) < K, other=0.0)
    v = tl.load(p_v, mask=(i_v * BV + tl.arange(0, BV)) < V, other=0.0)

    # Element-wise operations
    qk = q * k * scale
    o = tl.dot(qk, v)

    # Store result
    tl.store(p_o, o, mask=(i_v * BV + tl.arange(0, BV)) < V)

@triton.jit
def fused_recurrent_bwd_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr,
    scale, B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # Compute pointers
    p_q = q_ptr + i_bh * H * K + i_k * BK + tl.arange(0, BK)
    p_k = k_ptr + i_bh * H * K + i_k * BK + tl.arange(0, BK)
    p_v = v_ptr + i_bh * H * V + i_v * BV + tl.arange(0, BV)
    p_do = do_ptr + i_bh * H * V + i_v * BV + tl.arange(0, BV)

    # Load data
    q = tl.load(p_q, mask=(i_k * BK + tl.arange(0, BK)) < K, other=0.0)
    k = tl.load(p_k, mask=(i_k * BK + tl.arange(0, BK)) < K, other=0.0)
    v = tl.load(p_v, mask=(i_v * BV + tl.arange(0, BV)) < V, other=0.0)
    do = tl.load(p_do, mask=(i_v * BV + tl.arange(0, BV)) < V, other=0.0)

    # Compute gradients
    dq = tl.dot(do, v) * k * scale
    dk = tl.dot(do, q) * v * scale
    dv = tl.dot(q, k) * do * scale

    # Store gradients
    tl.store(dq_ptr + i_bh * H * K + i_k * BK + tl.arange(0, BK), dq, mask=(i_k * BK + tl.arange(0, BK)) < K)
    tl.store(dk_ptr + i_bh * H * K + i_k * BK + tl.arange(0, BK), dk, mask=(i_k * BK + tl.arange(0, BK)) < K)
    tl.store(dv_ptr + i_bh * H * V + i_v * BV + tl.arange(0, BV), dv, mask=(i_v * BV + tl.arange(0, BV)) < V)

class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BK, BV = min(K, 32), min(V, 32)

        o = torch.empty_like(v)
        grid = (triton.cdiv(V, BV), triton.cdiv(K, BK), B * H)
        fused_recurrent_fwd_kernel[grid](q, k, v, o, scale, B=B, H=H, T=T, K=K, V=V, BK=BK, BV=BV)

        ctx.save_for_backward(q, k, v, scale)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, scale = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        BK, BV = min(K, 32), min(V, 32)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        grid = (triton.cdiv(V, BV), triton.cdiv(K, BK), B * H)
        fused_recurrent_bwd_kernel[grid](q, k, v, do, dq, dk, dv, scale, B=B, H=H, T=T, K=K, V=V, BK=BK, BV=BV)

        return dq, dk, dv, None

def fused_recurrent_delta_rule(q, k, v, scale):
    return FusedRecurrentFunction.apply(q, k, v, scale)
