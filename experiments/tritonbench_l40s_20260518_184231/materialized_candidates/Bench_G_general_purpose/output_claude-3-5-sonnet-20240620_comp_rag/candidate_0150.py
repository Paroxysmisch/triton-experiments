import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,  # Pointer to output tensor
    input_ptr,   # Pointer to input tensor
    input_row_stride,  # Stride for input rows
    output_row_stride, # Stride for output rows
    n_rows,      # Number of rows in input
    n_cols,      # Number of columns in input
    BLOCK_SIZE: tl.constexpr,  # Static block size for processing
    num_stages: tl.constexpr,  # Number of pipeline stages
):
    # Get the program ID and total number of programs
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    
    # Process rows in a strided fashion
    for row_idx in range(pid, n_rows, grid_size):
        # Compute pointers for current row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        
        # Create offsets for columns
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        
        # Load input row with masking for out-of-bounds access
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        
        # Compute max for numerical stability
        row_max = tl.max(row, axis=0)
        
        # Compute exponentials with stability correction
        row_minus_max = row - row_max
        numerator = tl.exp(row_minus_max)
        
        # Compute sum for normalization
        denominator = tl.sum(numerator, axis=0)
        
        # Compute softmax output
        softmax_output = numerator / denominator
        
        # Store results
        output_row_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

# Wrapper function
def softmax(x):
    # Input validation
    assert x.dim() == 2, "Input tensor must be 2-dimensional"
    assert x.is_cuda, "Input tensor must be on GPU"
    
    # Get dimensions
    n_rows, n_cols = x.shape
    
    # Compute block size (next power of 2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Configure grid
    num_warps = 4
    num_stages = 2
    
    # Launch kernel
    grid = (triton.cdiv(n_rows, num_warps),)  # Ensure enough threads for all rows
    
    softmax_kernel[grid](
        output_ptr=output.data_ptr(),
        input_ptr=x.data_ptr(),
        input_row_stride=x.stride(0),
        output_row_stride=output.stride(0),
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
    )
    
    return output
