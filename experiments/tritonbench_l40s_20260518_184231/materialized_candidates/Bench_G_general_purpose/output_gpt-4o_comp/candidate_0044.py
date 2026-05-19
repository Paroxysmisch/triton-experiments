import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr, stride, n_cols, BLOCK_SIZE: tl.constexpr
):
    # Get the row index for the current program instance
    row_idx = tl.program_id(0)
    
    # Calculate the starting index for this row
    row_start = row_idx * stride
    
    # Create a range of indices for this row
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    
    # Load the data for this row
    row_data = tl.load(input_ptr + offsets, mask=offsets < n_cols, other=-float('inf'))
    
    # Compute the maximum value in the row for numerical stability
    row_max = tl.max(row_data, axis=0)
    
    # Subtract the max and exponentiate
    row_data = tl.exp(row_data - row_max)
    
    # Compute the sum of exponentiated values
    row_sum = tl.sum(row_data, axis=0)
    
    # Normalize the row by dividing by the sum
    softmax_row = row_data / row_sum
    
    # Store the result back to the output pointer
    tl.store(output_ptr + offsets, softmax_row, mask=offsets < n_cols)

def softmax(input_tensor):
    # Ensure the input is a 2D tensor
    assert input_tensor.dim() == 2, "Input must be a 2D tensor"
    
    # Get the dimensions of the input tensor
    n_rows, n_cols = input_tensor.shape
    
    # Determine the BLOCK_SIZE as the next power of two of n_cols
    BLOCK_SIZE = 2**((n_cols - 1).bit_length())
    
    # Adjust the number of warps for larger block sizes
    num_warps = min(4, (BLOCK_SIZE + 31) // 32)
    
    # Allocate an output tensor
    output_tensor = torch.empty_like(input_tensor)
    
    # Define the grid size
    grid = (n_rows,)
    
    # Launch the Triton kernel
    softmax_kernel[grid](
        input_tensor, output_tensor, input_tensor.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_tensor

# Example usage:
input_tensor = torch.randn(128, 512, device='cuda')  # Example input
output_tensor = softmax(input_tensor)
print(output_tensor)
