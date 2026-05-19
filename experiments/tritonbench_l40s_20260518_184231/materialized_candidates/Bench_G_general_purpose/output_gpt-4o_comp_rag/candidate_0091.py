import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd, custom_bwd

@triton.jit
def fused_recurrent_fwd_kernel(
    q, k, v, o, initial_state, final_state, scale, 
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, 
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, 
    BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, 
    STORE_FINAL_STATE: tl.constexpr, REVERSE: tl.constexpr
):
    # Get the program IDs
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # Calculate pointers for q, k, v, and o
    p_q = q + i_bh * T * K + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_k = k + i_bh * T * K + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_v = v + i_bh * T * V + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_o = o + i_bh * T * V + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    
    # Initialize hidden state
    h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = initial_state + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_h0, mask=True, other=0).to(tl.float32)

    # Process the sequence
    for _ in range(T):
        b_q = tl.load(p_q, mask=True, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=True, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=True, other=0).to(tl.float32)
        
        # Update hidden state and output
        h += b_k[None, :] * b_v[:, None]
        b_o = h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=True)
        
        # Move pointers
        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_v += -V if REVERSE else V
        p_o += -V if REVERSE else V

    # Store final state if required
    if STORE_FINAL_STATE:
        p_ht = final_state + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, h.to(p_ht.dtype.element_ty), mask=True)

@triton.jit
def fused_recurrent_bwd_kernel(
    q, k, v, do, dq, dk, dv, initial_state, scale, 
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, 
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, 
    BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, 
    REVERSE: tl.constexpr
):
    # Get the program IDs
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # Calculate pointers for q, k, v, and do
    p_q = q + i_bh * T * K + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_k = k + i_bh * T * K + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_v = v + i_bh * T * V + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_do = do + i_bh * T * V + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    
    # Initialize gradients
    h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = initial_state + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_h0, mask=True, other=0).to(tl.float32)

    # Process the sequence
    for _ in range(T):
        b_k = tl.load(p_k, mask=True, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=True, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=True, other=0).to(tl.float32)
        
        # Update hidden state and gradients
        h += b_k[:, None] * b_v[None, :]
        b_dq = tl.sum(h * b_do[None, :], axis=1) * scale
        tl.store(dq + i_bh * T * K + i_k * BK + tl.arange(0, BK), b_dq.to(dq.dtype.element_ty), mask=True)

        # Move pointers
        p_k += -K if REVERSE else K
        p_v += -V if REVERSE else V
        p_do += -V if REVERSE else V

# Define the PyTorch function class
class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, q, k, v, scale=None, initial_state=None, output_final_state=False, reverse=False):
        B, H, T, K, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[3]
        if scale is None:
            scale = K ** -0.5

        BK, BV = min(K, 32), min(V, 32)
        grid = (triton.cdiv(V, BV), triton.cdiv(K, BK), B * H)

        final_state = q.new_empty(B, H, K, V) if output_final_state else None

        o = q.new_empty(B, H, T, V, dtype=torch.float32)
        fused_recurrent_fwd_kernel[grid](
            q, k, v, o, initial_state, final_state, scale,
            B=B, H=H, T=T, K=K, V=V, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            REVERSE=reverse
        )
        ctx.save_for_backward(q, k, v, initial_state)
        ctx.scale = scale
        ctx.reverse = reverse
        return o, final_state

    @staticmethod
    @custom_bwd
    def backward(ctx, do, d_final_state=None):
        q, k, v, initial_state = ctx.saved_tensors
        B, H, T, K, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[3]
        scale = ctx.scale

        BK, BV = min(K, 32), min(V, 32)
        grid = (triton.cdiv(V, BV), triton.cdiv(K, BK), B * H)

        dq = q.new_empty(B, H, T, K, dtype=torch.float32)
        dk = k.new_empty(B, H, T, K, dtype=torch.float32)
        dv = v.new_empty(B, H, T, V, dtype=torch.float32)

        fused_recurrent_bwd_kernel[grid](
            q, k, v, do, dq, dk, dv, initial_state, scale,
            B=B, H=H, T=T, K=K, V=V, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            REVERSE=ctx.reverse
        )
        return dq, dk, dv, None, None, None, None

def fused_recurrent_delta_rule(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
