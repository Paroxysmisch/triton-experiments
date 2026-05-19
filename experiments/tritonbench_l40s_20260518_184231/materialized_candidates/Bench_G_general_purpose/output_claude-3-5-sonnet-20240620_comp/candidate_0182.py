import triton
import triton.language as tl
import torch

@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    # Pointers to tensors
    x_ptr, g_ptr, o_ptr, h0_ptr, ht_ptr,
    # Tensor strides
    stride_x_t, stride_x_b, stride_x_d,
    stride_g_t, stride_g_b, stride_g_d,
    stride_o_t, stride_o_b, stride_o_d,
    # Constants
    T, D, BD,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    batch_idx = pid
    
    # Initialize hidden state
    if USE_INITIAL_STATE:
        h = tl.load(h0_ptr + batch_idx * D + tl.arange(0, BLOCK_SIZE))
    else:
        h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Main recurrent loop
    for t in range(T):
        # Load inputs for current timestep
        offs_x = batch_idx * stride_x_b + t * stride_x_t + tl.arange(0, BLOCK_SIZE)
        offs_g = batch_idx * stride_g_b + t * stride_g_t + tl.arange(0, BLOCK_SIZE)
        offs_o = batch_idx * stride_o_b + t * stride_o_t + tl.arange(0, BLOCK_SIZE)
        
        x = tl.load(x_ptr + offs_x)
        g = tl.load(g_ptr + offs_g)
        
        # Update hidden state: h = g * h + x
        h = g * h + x
        
        # Store output
        tl.store(o_ptr + offs_o, h)
    
    # Store final state if requested
    if STORE_FINAL_STATE:
        tl.store(ht_ptr + batch_idx * D + tl.arange(0, BLOCK_SIZE), h)


@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    # Pointers to tensors
    dx_ptr, dg_ptr, do_ptr, x_ptr, g_ptr, o_ptr,
    # Tensor strides
    stride_x_t, stride_x_b, stride_x_d,
    stride_g_t, stride_g_b, stride_g_d,
    stride_o_t, stride_o_b, stride_o_d,
    # Constants
    T, D, BD,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    batch_idx = pid
    
    # Initialize gradient accumulator
    dh = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Backward pass through time
    for t in range(T-1, -1, -1):
        # Load gradients and states
        offs_x = batch_idx * stride_x_b + t * stride_x_t + tl.arange(0, BLOCK_SIZE)
        offs_g = batch_idx * stride_g_b + t * stride_g_t + tl.arange(0, BLOCK_SIZE)
        offs_o = batch_idx * stride_o_b + t * stride_o_t + tl.arange(0, BLOCK_SIZE)
        
        do = tl.load(do_ptr + offs_o)
        g = tl.load(g_ptr + offs_g)
        o = tl.load(o_ptr + offs_o)
        
        # Accumulate gradients
        dh = dh + do
        
        # Compute input gradients
        dx = dh
        dg = dh * o
        
        # Store gradients
        tl.store(dx_ptr + offs_x, dx)
        tl.store(dg_ptr + offs_g, dg)
        
        # Update dh for next timestep
        dh = dh * g

class FusedRecurrentHGRNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, h0=None, store_final_state=False):
        T, B, D = x.shape
        
        # Allocate output tensor
        o = torch.empty_like(x)
        ht = torch.empty((B, D), device=x.device) if store_final_state else None
        
        # Configure grid
        grid = (B,)
        
        # Save tensors for backward
        ctx.save_for_backward(x, g, o)
        
        # Launch kernel
        fused_recurrent_hgrn_fwd_kernel[grid](
            x, g, o, h0 if h0 is not None else x.new_zeros((1,)),
            ht if ht is not None else x.new_zeros((1,)),
            x.stride(0), x.stride(1), x.stride(2),
            g.stride(0), g.stride(1), g.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            T, D, B * D,
            h0 is not None,
            store_final_state,
            BLOCK_SIZE=min(D, 1024),
        )
        
        return (o, ht) if store_final_state else o
    
    @staticmethod
    def backward(ctx, do, dht=None):
        x, g, o = ctx.saved_tensors
        T, B, D = x.shape
        
        # Allocate gradient tensors
        dx = torch.empty_like(x)
        dg = torch.empty_like(g)
        
        # Configure grid
        grid = (B,)
        
        # Launch kernel
        fused_recurrent_hgrn_bwd_kernel[grid](
            dx, dg, do, x, g, o,
            x.stride(0), x.stride(1), x.stride(2),
            g.stride(0), g.stride(1), g.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            T, D, B * D,
            BLOCK_SIZE=min(D, 1024),
        )
        
        return dx, dg, None, None


def fused_recurrent_hgrn(x, g, h0=None, store_final_state=False):
    """
    Fused Recurrent Hierarchical Gated Recurrent Network
    
    Args:
        x: Input tensor of shape (T, B, D)
        g: Gates tensor of shape (T, B, D)
        h0: Optional initial hidden state of shape (B, D)
        store_final_state: Whether to return the final hidden state
    
    Returns:
        o: Output tensor of shape (T, B, D)
        ht: Final hidden state of shape (B, D) if store_final_state=True
    """
    return FusedRecurrentHGRNFunction.apply(
        x.detach(), g.detach(),
        h0.detach() if h0 is not None else None,
        store_final_state
    )
