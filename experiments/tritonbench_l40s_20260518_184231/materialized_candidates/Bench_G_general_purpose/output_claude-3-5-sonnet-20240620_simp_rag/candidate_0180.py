import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    x_ptr, g_ptr, o_ptr, h0_ptr, ht_ptr,
    T: tl.constexpr, D: tl.constexpr, 
    BD: tl.constexpr = 32,  # Block dimension, tunable
    USE_INITIAL_STATE: tl.constexpr = True,
    STORE_FINAL_STATE: tl.constexpr = False
):
    # Get program ID for parallel execution
    pid_d = tl.program_id(0)  # Dimension parallel
    pid_bh = tl.program_id(1)  # Batch-Head parallel
    
    # Calculate offsets
    offs_d = pid_d * BD + tl.arange(0, BD)
    mask = offs_d < D
    
    # Calculate base pointers
    base_offset = pid_bh * T * D + offs_d
    x_block_ptr = x_ptr + base_offset
    g_block_ptr = g_ptr + base_offset
    o_block_ptr = o_ptr + base_offset
    
    # Initialize hidden state
    h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        h0_block_ptr = h0_ptr + pid_bh * D + offs_d
        h = tl.load(h0_block_ptr, mask=mask)
    
    # Forward pass through time steps
    for t in range(T):
        x = tl.load(x_block_ptr + t * D, mask=mask)
        g = tl.load(g_block_ptr + t * D, mask=mask)
        
        # Core HGRN computation
        h = tl.exp(g) * h + x
        tl.store(o_block_ptr + t * D, h, mask=mask)
    
    # Store final state if requested
    if STORE_FINAL_STATE:
        ht_block_ptr = ht_ptr + pid_bh * D + offs_d
        tl.store(ht_block_ptr, h, mask=mask)

@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    g_ptr, o_ptr, dx_ptr, dg_ptr, do_ptr, h0_ptr,
    T: tl.constexpr, D: tl.constexpr,
    BD: tl.constexpr = 32,
    USE_INITIAL_STATE: tl.constexpr = True
):
    # Get program ID for parallel execution
    pid_d = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Calculate offsets
    offs_d = pid_d * BD + tl.arange(0, BD)
    mask = offs_d < D
    
    # Calculate base pointers
    base_offset = pid_bh * T * D + offs_d
    g_block_ptr = g_ptr + base_offset
    o_block_ptr = o_ptr + base_offset
    dx_block_ptr = dx_ptr + base_offset
    dg_block_ptr = dg_ptr + base_offset
    do_block_ptr = do_ptr + base_offset
    
    # Initialize gradient accumulators
    dh = tl.zeros([BD], dtype=tl.float32)
    
    # Backward pass through time steps
    for t in range(T-1, -1, -1):
        do = tl.load(do_block_ptr + t * D, mask=mask)
        g = tl.load(g_block_ptr + t * D, mask=mask)
        
        # Load previous hidden state
        if t > 0:
            prev_h = tl.load(o_block_ptr + (t-1) * D, mask=mask)
        elif USE_INITIAL_STATE:
            h0_block_ptr = h0_ptr + pid_bh * D + offs_d
            prev_h = tl.load(h0_block_ptr, mask=mask)
        else:
            prev_h = tl.zeros([BD], dtype=tl.float32)
        
        # Backward pass computations
        dh = dh + do
        dx = dh
        dh = dh * tl.exp(g)
        dg = dh * prev_h
        
        # Store gradients
        tl.store(dx_block_ptr + t * D, dx, mask=mask)
        tl.store(dg_block_ptr + t * D, dg, mask=mask)

class FusedRecurrentHGRNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, initial_state=None, output_final_state=False):
        B, H, T, D = x.shape
        o = torch.empty_like(x)
        final_state = torch.empty((B, H, D), device=x.device) if output_final_state else None
        
        grid = (triton.cdiv(D, 32), B * H)
        fused_recurrent_hgrn_fwd_kernel[grid](
            x, g, o, initial_state, final_state,
            T=T, D=D,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state
        )
        
        ctx.save_for_backward(g, o, initial_state)
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht=None):
        g, o, initial_state = ctx.saved_tensors
        B, H, T, D = do.shape
        
        dx = torch.empty_like(do)
        dg = torch.empty_like(g)
        
        grid = (triton.cdiv(D, 32), B * H)
        fused_recurrent_hgrn_bwd_kernel[grid](
            g, o, dx, dg, do, initial_state,
            T=T, D=D,
            USE_INITIAL_STATE=initial_state is not None
        )
        
        return dx, dg, None, None

def fused_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Recurrent HGRN operation.
    
    Args:
        x: Input tensor of shape (B, H, T, D)
        g: Gate tensor of shape (B, H, T, D)
        initial_state: Optional initial hidden state of shape (B, H, D)
        output_final_state: Whether to return the final hidden state
        
    Returns:
        Tuple of (output tensor, final hidden state if requested)
    """
    if initial_state is not None:
        initial_state = initial_state.detach()
    return FusedRecurrentHGRNFunction.apply(x, g, initial_state, output_final_state)
