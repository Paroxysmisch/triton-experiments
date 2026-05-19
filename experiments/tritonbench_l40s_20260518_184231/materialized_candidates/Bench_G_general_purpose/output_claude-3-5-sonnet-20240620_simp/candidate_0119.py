import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(
    output_ptr,  # Pointer to output tensor
    input_ptr,   # Pointer to input tensor
    input_row_stride,  # Stride between rows of input
    output_row_stride, # Stride between rows of output
    n_cols,      # Number of columns in input
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Get the row index
    row_idx = tl.program_id(axis=0)
    
    # Compute pointers to row start
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Create offsets for this block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row into SRAM
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Compute exponentials with stability
    numerator = tl.exp(row - row_max)
    
    # Compute sum for denominator
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize
    softmax_output = numerator / denominator
    
    # Store the result
    tl.store(out_row_start_ptr + col_offsets, softmax_output, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    # Get input dimensions
    n_rows, n_cols = x.shape
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Determine block size (round up to multiple of 32 for efficiency)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 2048)  # Cap block size
    
    # Launch kernel
    grid = (n_rows,)  # One block per row
    
    softmax_kernel[grid](
        output_ptr=output,
        input_ptr=x,
        input_row_stride=x.stride(0),
        output_row_stride=output.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
