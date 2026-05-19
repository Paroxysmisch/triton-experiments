import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd
from typing import Optional, Tuple

@triton.jit
def fused_recurrent_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, o_ptr, h0_ptr, ht_ptr,
    # Matrix dimensions
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    D: tl.constexpr, # Hidden dimension
    BLOCK_SIZE: tl.constexpr,
    # Other parameters
    scale, beta,
    stride_qb, stride_kb, stride_vb,
    stride_ob, stride_h,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    # Batch and head indices
    batch_idx = pid // H
    head_idx = pid % H
    
    # Initialize offsets
    offs_q = batch_idx * stride_qb + head_idx * stride_h
    offs_k = batch_idx * stride_kb + head_idx * stride_h
    offs_v = batch_idx * stride_vb + head_idx * stride_h
    offs_o = batch_idx * stride_ob + head_idx * stride_h
    
    # Load block
    offs_m = tl.arange(0, BLOCK_SIZE)
    
    # Initialize hidden state
    h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if USE_INITIAL_STATE:
        h0_off = batch_idx * D + head_idx * BLOCK_SIZE + offs_m
        h = tl.load(h0_ptr + h0_off, mask=offs_m < D, other=0.0)
    
    # Main loop
    for t in range(T):
        # Load inputs
        q = tl.load(q_ptr + offs_q + t * D + offs_m, mask=offs_m < D, other=0.0)
        k = tl.load(k_ptr + offs_k + t * D + offs_m, mask=offs_m < D, other=0.0)
        v = tl.load(v_ptr + offs_v + t * D + offs_m, mask=offs_m < D, other=0.0)
        
        # Update hidden state
        h = beta * h + k * v
        
        # Compute output
        o = scale * q * h
        
        # Store output
        tl.store(o_ptr + offs_o + t * D + offs_m, o, mask=offs_m < D)
    
    # Store final state if needed
    if STORE_FINAL_STATE:
        ht_off = batch_idx * D + head_idx * BLOCK_SIZE + offs_m
        tl.store(ht_ptr + ht_off, h, mask=offs_m < D)

@triton.jit
def fused_recurrent_bwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, do_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    h0_ptr,
    # Matrix dimensions
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    D: tl.constexpr, # Hidden dimension
    BLOCK_SIZE: tl.constexpr,
    # Other parameters
    scale, beta,
    stride_qb, stride_kb, stride_vb,
    stride_ob, stride_h,
    USE_INITIAL_STATE: tl.constexpr
):
    # Similar structure to forward kernel with backward pass logic
    pid = tl.program_id(0)
    batch_idx = pid // H
    head_idx = pid % H
    
    offs_m = tl.arange(0, BLOCK_SIZE)
    
    # Initialize gradients
    dh = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Backward pass
    for t in range(T-1, -1, -1):
        q = tl.load(q_ptr + offs_q + t * D + offs_m, mask=offs_m < D, other=0.0)
        k = tl.load(k_ptr + offs_k + t * D + offs_m, mask=offs_m < D, other=0.0)
        v = tl.load(v_ptr + offs_v + t * D + offs_m, mask=offs_m < D, other=0.0)
        do = tl.load(do_ptr + offs_o + t * D + offs_m, mask=offs_m < D, other=0.0)
        
        # Compute gradients
        dq = scale * dh * k
        dk = scale * dh * q
        dv = scale * dh * q * k
        
        # Update dh
        dh = beta * dh + do
        
        # Store gradients
        tl.store(dq_ptr + offs_q + t * D + offs_m, dq, mask=offs_m < D)
        tl.store(dk_ptr + offs_k + t * D + offs_m, dk, mask=offs_m < D)
        tl.store(dv_ptr + offs_v + t * D + offs_m, dv, mask=offs_m < D)

class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, q, k, v, beta=1.0, scale=None, initial_state=None):
        # Save context
        ctx.save_for_backward(q, k, v, initial_state)
        ctx.beta = beta
        ctx.scale = scale if scale is not None else 1.0 / (q.shape[-1] ** 0.5)
        
        # Output tensor
        o = torch.empty_like(q)
        
        # Launch kernel
        grid = (q.shape[0] * q.shape[1],)  # batch * heads
        fused_recurrent_fwd_kernel[grid](
            q, k, v, o, initial_state, None,
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],
            min(256, q.shape[3]),
            ctx.scale, ctx.beta,
            q.stride(0), k.stride(0), v.stride(0),
            o.stride(0), q.stride(1),
            initial_state is not None,
            False
        )
        return o

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        q, k, v, initial_state = ctx.saved_tensors
        
        # Initialize gradient tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Launch kernel
        grid = (q.shape[0] * q.shape[1],)  # batch * heads
        fused_recurrent_bwd_kernel[grid](
            q, k, v, grad_output,
            dq, dk, dv, initial_state,
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],
            min(256, q.shape[3]),
            ctx.scale, ctx.beta,
            q.stride(0), k.stride(0), v.stride(0),
            grad_output.stride(0), q.stride(1),
            initial_state is not None
        )
        return dq, dk, dv, None, None, None

def fused_recurrent(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   beta: float = 1.0,
                   scale: Optional[float] = None,
                   initial_state: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Fused recurrent operation.
    
    Args:
        q: Query tensor of shape (batch, heads, time, dims)
        k: Key tensor of shape (batch, heads, time, dims)
        v: Value tensor of shape (batch, heads, time, dims)
        beta: Decay factor for hidden state
        scale: Scaling factor for attention scores
        initial_state: Optional initial hidden state
    
    Returns:
        Output tensor of shape (batch, heads, time, dims)
    """
    return FusedRecurrentFunction.apply(q, k, v, beta, scale, initial_state)
