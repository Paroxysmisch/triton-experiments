import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    row_idx = tl.program_id(0)
    
    # Compute pointers to input/output row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Initialize local variables
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input row
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract max from row and compute exponentials
    row = row - row_max
    numerator = tl.exp(row)
    
    # Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize
    softmax_output = numerator / denominator
    
    # Store result
    tl.store(out_row_start_ptr + col_offsets, softmax_output, mask=mask)

def softmax(x):
    # Get input dimensions
    n_rows, n_cols = x.shape
    
    # Determine block size (round up to multiple of 32 for efficiency)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Maximum block size
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Determine number of warps based on block size
    num_warps = 4
    if BLOCK_SIZE >= 512:
        num_warps = 8
    if BLOCK_SIZE >= 768:
        num_warps = 16
        
    # Enqueue kernel
    softmax_kernel[(n_rows,)](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return output
