import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,  # Pointer to output tensor
    input_ptr,   # Pointer to input tensor
    input_row_stride,  # Stride between rows of input
    output_row_stride, # Stride between rows of output
    n_rows,      # Number of rows in the matrix
    n_cols,      # Number of columns in the matrix
    BLOCK_SIZE: tl.constexpr,  # Static block size for optimization
):
    # Get the program ID for the current row
    row_idx = tl.program_id(0)
    
    # Compute pointers to the row we'll process
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Create a range for column indices
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Compute pointers for each element in the row
    input_ptrs = row_start_ptr + col_offsets
    
    # Create mask for valid columns (in case n_cols < BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row elements from DRAM into SRAM
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Compute softmax numerator
    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)
    
    # Compute softmax denominator
    denominator = tl.sum(numerator, axis=0)
    
    # Compute softmax output
    softmax_output = numerator / denominator
    
    # Write output back to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x):
    # Extract input dimensions
    n_rows, n_cols = x.shape
    
    # Calculate optimal block size (next power of 2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Determine number of warps based on block size
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 4
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Configure grid for parallel execution
    grid = (n_rows,)
    
    # Launch kernel
    softmax_kernel[grid](
        y,                  # output tensor
        x,                  # input tensor
        x.stride(0),       # input row stride
        y.stride(0),       # output row stride
        n_rows,            # number of rows
        n_cols,            # number of columns
        BLOCK_SIZE=BLOCK_SIZE,  # block size for processing
        num_warps=num_warps,    # number of warps
    )
    
    return y

# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(1823, 781, device='cuda')
    y_triton = softmax(x)
