import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(
    output_ptr, input_ptr,
    n_rows, n_cols,
    row_stride_in, row_stride_out,
    TILE_N: tl.constexpr
):
    # Get the program ID for the current row
    row_idx = tl.program_id(0)
    
    # Compute pointers for current row
    row_start_in = input_ptr + row_idx * row_stride_in
    row_start_out = output_ptr + row_idx * row_stride_out
    
    # Create offsets for the current tile
    col_offsets = tl.arange(0, TILE_N)
    
    # Initialize variables for max and sum computation
    max_val = -float('inf')
    sum_exp = 0.0
    
    # Phase 1: Find max value for numerical stability
    for tile_start in range(0, n_cols, TILE_N):
        # Compute mask for valid elements
        mask = col_offsets + tile_start < n_cols
        # Load input values for current tile
        x = tl.load(row_start_in + tile_start + col_offsets, mask=mask, other=-float('inf'))
        # Update running maximum
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    
    # Phase 2: Compute sum of exponentials
    for tile_start in range(0, n_cols, TILE_N):
        mask = col_offsets + tile_start < n_cols
        x = tl.load(row_start_in + tile_start + col_offsets, mask=mask, other=-float('inf'))
        # Subtract max for numerical stability and compute exp
        x_stable = tl.exp(x - max_val)
        # Update running sum
        sum_exp += tl.sum(x_stable, axis=0)
    
    # Phase 3: Compute final softmax values and store
    for tile_start in range(0, n_cols, TILE_N):
        mask = col_offsets + tile_start < n_cols
        x = tl.load(row_start_in + tile_start + col_offsets, mask=mask, other=-float('inf'))
        # Compute stable softmax
        x_stable = tl.exp(x - max_val) / sum_exp
        # Store results
        tl.store(row_start_out + tile_start + col_offsets, x_stable, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute row-wise softmax using Triton kernel.
    
    Args:
        x: Input tensor of shape (M, N)
    Returns:
        Tensor of shape (M, N) containing row-wise softmax probabilities
    """
    n_rows, n_cols = x.shape
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Compute optimal tile size (power of 2)
    TILE_N = min(triton.next_power_of_2(n_cols), 512)
    
    # Configure grid and block sizes
    num_warps = 4
    if TILE_N >= 256:
        num_warps = 8
    if TILE_N >= 512:
        num_warps = 16
    
    # Launch kernel
    softmax_kernel_online_v2[(n_rows,)](
        output_ptr=output,
        input_ptr=x,
        n_rows=n_rows,
        n_cols=n_cols,
        row_stride_in=x.stride(0),
        row_stride_out=output.stride(0),
        TILE_N=TILE_N,
        num_warps=num_warps
    )
    
    return output
