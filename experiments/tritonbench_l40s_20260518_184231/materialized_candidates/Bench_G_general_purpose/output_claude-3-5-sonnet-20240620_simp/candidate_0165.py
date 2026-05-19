import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,          # Pointer to output tensor
    input_ptr,           # Pointer to input tensor
    mask_ptr,           # Pointer to optional attention mask
    row_stride,         # Stride between rows
    n_cols,             # Number of columns
    BLOCK_SIZE: tl.constexpr  # Block size for parallel processing
):
    # Get the program ID
    row_idx = tl.program_id(0)
    
    # Compute memory offsets for this row
    row_start_ptr = input_ptr + row_idx * row_stride
    mask_row_start_ptr = mask_ptr + row_idx * row_stride if mask_ptr is not None else None
    
    # Load row elements using block-based memory access
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Apply attention mask if provided
    if mask_ptr is not None:
        attention_mask = tl.load(mask_row_start_ptr + col_offsets, mask=mask, other=0)
        row = row * attention_mask
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Compute exponentials
    row = tl.exp(row - row_max)
    
    # Compute sum for normalization
    row_sum = tl.sum(row, axis=0)
    
    # Normalize
    row = row / row_sum
    
    # Store the result
    output_row_start_ptr = output_ptr + row_idx * row_stride
    tl.store(output_row_start_ptr + col_offsets, row, mask=mask)

def softmax(x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Apply softmax to input tensor along the last dimension.
    
    Args:
        x: Input tensor of shape (..., n)
        mask: Optional attention mask of same shape as x
    
    Returns:
        Output tensor of same shape as input with softmax applied
    """
    # Handle input validation and reshaping
    orig_shape = x.shape
    if len(x.shape) > 2:
        x = x.reshape(-1, x.shape[-1])
    
    # Get tensor dimensions
    n_rows, n_cols = x.shape
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Configure block size and grid
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    
    # Launch kernel
    softmax_kernel[grid](
        output,
        x,
        mask if mask is not None else x.new_empty(0),  # Empty tensor if no mask
        x.stride(0),
        n_cols,
        BLOCK_SIZE,
    )
    
    # Restore original shape if needed
    if len(orig_shape) > 2:
        output = output.reshape(orig_shape)
    
    return output
