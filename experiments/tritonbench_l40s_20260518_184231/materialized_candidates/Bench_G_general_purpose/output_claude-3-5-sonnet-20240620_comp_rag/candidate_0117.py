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
    BLOCK_SIZE: tl.constexpr,  # Size of parallel processing block
):
    # Get the row index we're processing
    row_idx = tl.program_id(axis=0)
    
    # Calculate pointers to start of current row in input and output
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Create a range for column indices
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Create mask for valid columns (less than n_cols)
    mask = col_offsets < n_cols
    
    # Load input row into SRAM, replacing invalid values with -inf
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Find maximum value for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract max and compute exponentials (numerator)
    numerator = tl.exp(row - row_max)
    
    # Compute sum for normalization (denominator)
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize to get softmax values
    softmax_output = numerator / denominator
    
    # Store results back to global memory
    tl.store(out_row_start_ptr + col_offsets, softmax_output, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax using Triton kernel.
    
    Args:
        x: Input tensor of shape (n_rows, n_cols)
    Returns:
        Output tensor of same shape with softmax applied to each row
    """
    n_rows, n_cols = x.shape
    # Create output tensor with same dtype and device as input
    output = torch.empty_like(x)
    
    # Calculate block size as next power of 2, capped at 1024
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)
    
    # Launch kernel with one thread per row
    grid = (n_rows,)
    softmax_kernel[grid](
        output, x,
        x.stride(0), output.stride(0),
        n_cols, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output
