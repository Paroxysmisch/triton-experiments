import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,          # Pointer to output tensor
    input_ptr,           # Pointer to input tensor
    input_row_stride,    # Stride between rows in input tensor
    output_row_stride,   # Stride between rows in output tensor
    n_cols,              # Number of columns in input
    BLOCK_SIZE: tl.constexpr,  # Size of parallel block processing
):
    # Get the row index from program ID
    row_idx = tl.program_id(axis=0)
    
    # Calculate pointers to start of current row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Create offset range for the block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Create mask for valid columns
    mask = col_offsets < n_cols
    
    # Load row into SRAM, mask out invalid columns with -inf
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Find maximum value in row for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Compute numerator: exp(x - max(x))
    numerator = tl.exp(row - row_max)
    
    # Compute denominator: sum(exp(x - max(x)))
    denominator = tl.sum(numerator, axis=0)
    
    # Compute softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    softmax_output = numerator / denominator
    
    # Store the result
    tl.store(out_row_start_ptr + col_offsets, softmax_output, mask=mask)

def triton_softmax(x):
    # Get input dimensions
    n_rows, n_cols = x.shape
    
    # Create output tensor with same type and device as input
    output = torch.empty_like(x)
    
    # Calculate block size (power of 2, max 1024)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    # Launch kernel with one thread per row
    grid = (n_rows,)
    softmax_kernel[grid](
        output,
        x,
        x.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
