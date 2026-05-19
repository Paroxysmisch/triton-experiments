import triton
import triton.language as tl
import torch

# Define the Triton kernel for softmax computation
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols, BLOCK_SIZE: tl.constexpr
):
    # Identify the row this program is responsible for
    row_idx = tl.program_id(0)
    
    # Compute the start pointer for the current row
    row_start = row_idx * input_row_stride
    
    # Create a range of indices for the block
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate column indices for the current block
    col_idx = row_start + offsets
    
    # Load the row into on-chip memory, applying masking for bounds
    mask = col_idx < row_start + n_cols
    row = tl.load(input_ptr + col_idx, mask=mask, other=-float('inf'))
    
    # Numerical stabilization: subtract the max value in the row
    row_max = tl.max(row, axis=0)
    row = row - row_max
    
    # Compute exponentials
    exp_row = tl.exp(row)
    
    # Normalize to get softmax probabilities
    sum_exp_row = tl.sum(exp_row, axis=0)
    softmax_row = exp_row / sum_exp_row
    
    # Store the result back to the output matrix
    tl.store(output_ptr + col_idx, softmax_row, mask=mask)

# Wrapper function to launch the kernel
def softmax(input_tensor):
    # Get the shape of the input tensor
    n_rows, n_cols = input_tensor.shape
    
    # Determine the optimal BLOCK_SIZE
    BLOCK_SIZE = 2 ** ((n_cols - 1).bit_length())
    
    # Adjust num_warps based on BLOCK_SIZE for efficiency
    num_warps = min(4, max(1, BLOCK_SIZE // 128))
    
    # Allocate output tensor
    output_tensor = torch.empty_like(input_tensor)
    
    # Launch the Triton kernel
    grid = (n_rows,)
    softmax_kernel[grid](
        input_tensor, output_tensor,
        input_tensor.stride(0), output_tensor.stride(0),
        n_cols, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return output_tensor

# Example usage
input_tensor = torch.randn(1024, 512, device='cuda')  # Example input tensor
output_tensor = softmax(input_tensor)
print(output_tensor)
