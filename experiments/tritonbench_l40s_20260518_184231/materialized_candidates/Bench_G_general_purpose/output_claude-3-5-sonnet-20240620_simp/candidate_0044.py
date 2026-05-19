import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,  # Pointer to output tensor
    input_ptr,   # Pointer to input tensor
    row_stride,  # Stride between rows
    n_cols,      # Number of columns
    BLOCK_SIZE: tl.constexpr  # Size of SIMD vector operations
):
    # Get the program ID
    row_idx = tl.program_id(0)
    
    # Compute memory offsets for this row
    row_start_ptr = input_ptr + row_idx * row_stride
    
    # Initialize variables for max computation
    row_max = -float('inf')
    
    # Load row data and compute max value
    for col in range(0, n_cols, BLOCK_SIZE):
        mask = col + tl.arange(0, BLOCK_SIZE) < n_cols
        block_ptr = row_start_ptr + col
        values = tl.load(block_ptr, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(values, axis=0))
    
    # Initialize sum for denominator
    exp_sum = 0.0
    
    # Compute exponentials and their sum
    for col in range(0, n_cols, BLOCK_SIZE):
        mask = col + tl.arange(0, BLOCK_SIZE) < n_cols
        block_ptr = row_start_ptr + col
        values = tl.load(block_ptr, mask=mask, other=-float('inf'))
        values = values - row_max  # Subtract max for numerical stability
        exp_values = tl.exp(values)
        exp_sum += tl.sum(exp_values * mask, axis=0)
    
    # Write normalized values to output
    output_row_start_ptr = output_ptr + row_idx * row_stride
    for col in range(0, n_cols, BLOCK_SIZE):
        mask = col + tl.arange(0, BLOCK_SIZE) < n_cols
        block_ptr = row_start_ptr + col
        values = tl.load(block_ptr, mask=mask, other=-float('inf'))
        values = values - row_max
        exp_values = tl.exp(values)
        normalized = exp_values / exp_sum
        output_block_ptr = output_row_start_ptr + col
        tl.store(output_block_ptr, normalized, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies softmax over the last dimension of the input tensor.
    
    Args:
        x: Input tensor of shape (..., n)
    Returns:
        Output tensor of same shape with softmax applied to last dimension
    """
    # Handle input tensor
    x_shape = x.shape
    n_rows = x.numel() // x_shape[-1]
    n_cols = x_shape[-1]
    x_reshaped = x.reshape(n_rows, n_cols)
    
    # Allocate output tensor
    output = torch.empty_like(x_reshaped)
    
    # Configure kernel parameters
    BLOCK_SIZE = 256  # Adjust based on hardware
    grid = (n_rows,)  # One thread block per row
    
    # Launch kernel
    softmax_kernel[grid](
        output,
        x_reshaped,
        x_reshaped.stride(0),
        n_cols,
        BLOCK_SIZE
    )
    
    # Reshape output back to original dimensions
    return output.reshape(x_shape)
