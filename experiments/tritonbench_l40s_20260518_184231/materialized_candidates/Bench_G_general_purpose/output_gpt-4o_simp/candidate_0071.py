import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def square_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_cols,  # Number of columns in the 2D tensor
    BLOCK_SIZE: tl.constexpr  # Size of the block
):
    # Get the program ID for the current block
    row_idx = tl.program_id(0)
    
    # Create a block of indices for this row
    col_indices = tl.arange(0, BLOCK_SIZE)
    
    # Compute the offset for this row
    row_offset = row_idx * n_cols
    
    # Compute the memory addresses for the input and output
    input_ptrs = input_ptr + row_offset + col_indices
    output_ptrs = output_ptr + row_offset + col_indices
    
    # Load a block of data from the input tensor
    input_vals = tl.load(input_ptrs, mask=col_indices < n_cols)
    
    # Compute the square of each element
    output_vals = input_vals * input_vals
    
    # Store the results back to the output tensor
    tl.store(output_ptrs, output_vals, mask=col_indices < n_cols)

# Define the wrapper function
def square(input_tensor):
    # Ensure the input tensor is a 2D tensor
    assert input_tensor.ndim == 2, "Input must be a 2D tensor"
    
    # Get the dimensions of the input tensor
    n_rows, n_cols = input_tensor.shape
    
    # Create an output tensor with the same shape
    output_tensor = torch.empty_like(input_tensor)
    
    # Determine block size and number of warps
    BLOCK_SIZE = 128  # This can be tuned for optimal performance
    num_warps = 4  # This can be tuned based on the hardware
    
    # Launch the Triton kernel
    grid = (n_rows,)  # One block per row
    square_kernel[grid](
        input_tensor,
        output_tensor,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_tensor

# Example usage
if __name__ == "__main__":
    # Create a sample input tensor
    input_tensor = torch.randn(1024, 512, device='cuda')
    
    # Compute the square using the Triton kernel
    output_tensor = square(input_tensor)
    
    # Print the result
    print(output_tensor)
