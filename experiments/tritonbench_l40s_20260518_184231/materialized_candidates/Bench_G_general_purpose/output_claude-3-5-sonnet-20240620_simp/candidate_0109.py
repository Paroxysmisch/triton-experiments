import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(
    output_ptr, input_ptr,
    M, N,
    TILE_N: tl.constexpr,
):
    # Program ID
    row_idx = tl.program_id(0)
    
    # Compute row offset
    row_start_ptr = input_ptr + row_idx * N
    
    # Initialize maximum value for numerical stability
    max_val = -float('inf')
    
    # First pass: find max value for numerical stability
    for tile_start_n in range(0, N, TILE_N):
        # Create block mask for bounds checking
        cols = tl.arange(0, TILE_N)
        mask = cols < (N - tile_start_n)
        
        # Load input values
        block_ptr = row_start_ptr + tile_start_n
        x = tl.load(block_ptr + cols, mask=mask, other=-float('inf'))
        
        # Update max value
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    
    # Second pass: compute exponentials and sum
    sum_exp = 0.0
    row_out_ptr = output_ptr + row_idx * N
    
    for tile_start_n in range(0, N, TILE_N):
        cols = tl.arange(0, TILE_N)
        mask = cols < (N - tile_start_n)
        
        # Load and compute exponentials
        block_ptr = row_start_ptr + tile_start_n
        x = tl.load(block_ptr + cols, mask=mask, other=-float('inf'))
        x = tl.exp(x - max_val)
        
        # Store intermediate results and accumulate sum
        block_out_ptr = row_out_ptr + tile_start_n
        tl.store(block_out_ptr + cols, x, mask=mask)
        sum_exp += tl.sum(x, axis=0)
    
    # Third pass: normalize by sum
    for tile_start_n in range(0, N, TILE_N):
        cols = tl.arange(0, TILE_N)
        mask = cols < (N - tile_start_n)
        
        block_out_ptr = row_out_ptr + tile_start_n
        x = tl.load(block_out_ptr + cols, mask=mask)
        x = x / sum_exp
        tl.store(block_out_ptr + cols, x, mask=mask)

def prev_multiple_of(a, b):
    return (a // b) * b

def softmax(x):
    """
    Compute softmax using Triton kernel
    Args:
        x: Input tensor of shape (M, N)
    Returns:
        Output tensor of shape (M, N) containing softmax probabilities
    """
    M, N = x.shape
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Configure TILE_N as power of 2, not exceeding N
    TILE_N = min(256, prev_multiple_of(N, 128))
    
    # Launch kernel
    grid = (M,)
    softmax_kernel_online_v2[grid](
        out, x,
        M, N,
        TILE_N=TILE_N,
    )
    return out
