import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(output_ptr, input_ptr, M, N, TILE_N: tl.constexpr):
    # Determine the row and column indices
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, TILE_N)
    
    # Calculate the starting pointers for the input and output
    row_start_ptr = input_ptr + row_idx * N
    output_row_start_ptr = output_ptr + row_idx * N
    
    # Load the tile from the input row
    input_ptrs = row_start_ptr + col_offsets
    row_tile = tl.load(input_ptrs, mask=col_offsets < N, other=-float('inf'))
    
    # Subtract the maximum value for numerical stability
    row_minus_max = row_tile - tl.max(row_tile, axis=0)
    
    # Exponentiate the stabilized values
    numerator = tl.exp(row_minus_max)
    
    # Sum the exponentiated values
    denominator = tl.sum(numerator, axis=0)
    
    # Compute the softmax
    softmax_output = numerator / denominator
    
    # Store the results back to the output
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < N)

def prev_multiple_of(a, b):
    return (a // b) * b

def softmax(x):
    M, N = x.shape
    # Determine the TILE_N as the largest power of 2 less than or equal to N
    TILE_N = triton.next_power_of_2(N)
    
    # Adjust TILE_N to be a previous multiple if needed
    TILE_N = prev_multiple_of(TILE_N, 2)
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Launch the kernel with one instance per row
    softmax_kernel_online_v2[(M,)](
        y,
        x,
        M,
        N,
        TILE_N=TILE_N
    )
    
    return y
