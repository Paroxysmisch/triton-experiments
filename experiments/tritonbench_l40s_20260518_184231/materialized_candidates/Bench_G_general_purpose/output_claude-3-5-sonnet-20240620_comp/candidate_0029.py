import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID (row index)
    row_idx = tl.program_id(0)
    
    # Calculate starting pointers for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Create a range for the block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid columns
    mask = col_offsets < n_cols
    
    # Load input row with masking
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Numerical stability: subtract max from row
    row_max = tl.max(row, axis=0)
    row = row - row_max
    
    # Compute exponentials
    numerator = tl.exp(row)
    
    # Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize to get softmax probabilities
    softmax_output = numerator / denominator
    
    # Store the result
    tl.store(out_row_start_ptr + col_offsets, softmax_output, mask=mask)

def softmax(x):
    # Get input dimensions
    n_rows, n_cols = x.shape
    
    # Calculate optimal block size (next power of 2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Adjust num_warps based on block size
    num_warps = 4 if BLOCK_SIZE <= 256 else 8
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate strides
    input_row_stride = x.stride(0)
    output_row_stride = output.stride(0)
    
    # Launch kernel with one block per row
    softmax_kernel[(n_rows,)](
        x,
        output,
        input_row_stride,
        output_row_stride,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return output
