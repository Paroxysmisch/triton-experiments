import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(
    output_ptr,  # Pointer to output matrix
    input_ptr,   # Pointer to input matrix
    input_row_stride,  # Stride between rows of input
    output_row_stride, # Stride between rows of output
    n_cols,      # Number of columns in matrix
    BLOCK_SIZE: tl.constexpr,  # Size of parallel block processing
):
    # Get the row index we're processing
    row_idx = tl.program_id(0)
    
    # Calculate start of the row in memory
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Create offsets for parallel column access
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load row data with mask for when BLOCK_SIZE > n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    # Convert to float32 for better numerical stability
    row_f32 = row.to(tl.float32)
    
    # Subtract max for numerical stability
    row_max = tl.max(row_f32, axis=0)
    row_minus_max = row_f32 - row_max
    
    # Compute exponentials
    numerator = tl.exp(row_minus_max)
    
    # Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize to get softmax values
    softmax_output = numerator / denominator
    
    # Store results back to memory
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output.to(row.dtype), mask=col_offsets < n_cols)

def softmax(input_tensor):
    """
    Wrapper function to compute softmax using the Triton kernel
    
    Args:
        input_tensor: Input tensor of shape (rows, cols)
    Returns:
        Output tensor with softmax applied to each row
    """
    # Get input dimensions
    n_rows, n_cols = input_tensor.shape
    
    # Determine block size (next power of 2 after n_cols)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Allocate output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate strides
    input_row_stride = input_tensor.stride(0)
    output_row_stride = output.stride(0)
    
    # Determine number of warps based on block size
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # Launch kernel
    softmax_kernel[(n_rows,)](
        output,
        input_tensor,
        input_row_stride,
        output_row_stride,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return output
