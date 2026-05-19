import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,  # Pointer to output tensor
    input_ptr,   # Pointer to input tensor
    row_stride,  # Stride between rows
    n_cols,      # Number of columns
    BLOCK_SIZE: tl.constexpr,  # Static block size parameter
):
    # Get the program ID
    row_idx = tl.program_id(0)
    
    # Compute pointer to row start
    row_start_ptr = input_ptr + row_idx * row_stride
    
    # Create offsets for the columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Create mask for valid columns
    mask = col_offsets < n_cols
    
    # Load row data
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract max and compute exponentials
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)
    
    # Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize to get softmax values
    softmax_output = numerator / denominator
    
    # Store results
    output_row_start_ptr = output_ptr + row_idx * row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax for each row of the input tensor using Triton.
    
    Args:
        x: Input tensor of shape (M, N) on GPU
    Returns:
        Output tensor of shape (M, N) containing softmax probabilities
    """
    n_rows, n_cols = x.shape
    
    # Ensure input is contiguous and on GPU
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.is_cuda, "Input tensor must be on GPU"
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Configure kernel parameters
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    
    # Launch kernel
    grid = (n_rows,)
    softmax_kernel[grid](
        output_ptr=output,
        input_ptr=x,
        row_stride=x.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return output

# Example usage
if __name__ == "__main__":
    # Create sample input
    x = torch.randn(1024, 256, device='cuda')
    
    # Run softmax
    y = softmax(x)
