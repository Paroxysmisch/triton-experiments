import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(
    output_ptr,
    input_ptr,
    M,
    N,
    input_row_stride,
    output_row_stride,
    TILE_N: tl.constexpr,
):
    # Get the current row index
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return  # Skip if out of bounds
    
    # Calculate pointers for the current row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Initialize max and sum for numerical stability
    current_max = tl.full((1,), -float('inf'), tl.float32)
    current_sum = tl.full((1,), 0.0, tl.float32)
    
    # Compute the number of tiles needed for this row
    num_tiles = tl.cdiv(N, TILE_N)
    
    # First pass: compute max and sum
    for tile_idx in range(num_tiles):
        tile_offset = tile_idx * TILE_N
        col_offsets = tile_offset + tl.arange(0, TILE_N)
        mask = col_offsets < N
        
        # Load tile from input
        row_vals = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        
        # Compute tile max and update current_max
        tile_max = tl.max(row_vals, axis=0)
        if tile_max > current_max:
            # Adjust sum based on new max
            scale = tl.exp(current_max - tile_max)
            current_sum *= scale
            current_max = tile_max
        
        # Compute exp and accumulate sum
        exp_vals = tl.exp(row_vals - current_max)
        tile_sum = tl.sum(exp_vals, axis=0)
        current_sum += tile_sum
    
    # Second pass: compute softmax and write output
    for tile_idx in range(num_tiles):
        tile_offset = tile_idx * TILE_N
        col_offsets = tile_offset + tl.arange(0, TILE_N)
        mask = col_offsets < N
        
        # Load tile again
        row_vals = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        
        # Compute softmax values
        exp_vals = tl.exp(row_vals - current_max)
        softmax_vals = exp_vals / current_sum
        
        # Store result
        tl.store(output_row_start_ptr + col_offsets, softmax_vals, mask=mask)

def softmax(x, TILE_N=128):
    M, N = x.shape
    y = torch.empty_like(x)
    grid = (M,)
    
    # Launch kernel
    softmax_kernel_online_v2[grid](
        y, x,
        M, N,
        x.stride(0),
        y.stride(0),
        TILE_N=TILE_N
    )
    return y
