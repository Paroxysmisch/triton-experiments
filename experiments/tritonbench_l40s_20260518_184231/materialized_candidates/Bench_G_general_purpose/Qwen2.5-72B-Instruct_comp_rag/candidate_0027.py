import triton
import triton.language as tl

# Define the softmax kernel
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Identify the current row using program ID
    row_idx = tl.program_id(0)
    
    # Calculate the starting pointer for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Define the block of threads
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load the row into on-chip memory with masking to handle cases where the block size exceeds column count
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_f32 = row.to(tl.float32)
    
    # Perform numerical stabilization by subtracting the maximum value from the row elements
    row_minus_max = row_f32 - tl.max(row_f32, axis=0)
    
    # Compute the exponentials
    numerator = tl.exp(row_minus_max)
    
    # Compute the denominator (sum of exponentials)
    denominator = tl.sum(numerator, axis=0)
    
    # Compute the softmax probabilities
    softmax_output = numerator / denominator
    
    # Write back the result to the output matrix
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output.to(row.dtype), mask=col_offsets < n_cols)

# Define the softmax function to facilitate kernel execution
def softmax(input_tensor, output_tensor):
    # Get the dimensions of the input tensor
    n_rows, n_cols = input_tensor.shape
    
    # Calculate the optimal BLOCK_SIZE
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE *= 2
    
    # Calculate the number of warps based on BLOCK_SIZE
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # Calculate the strides for row advancement
    input_row_stride = input_tensor.stride(0)
    output_row_stride = output_tensor.stride(0)
    
    # Launch the Triton kernel with one block per input matrix row
    softmax_kernel[(n_rows,)](output_tensor, input_tensor, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE, num_warps=num_warps)
    
    return output_tensor

# Example usage
import torch

# Create a random input tensor
input_tensor = torch.randn((1024, 1024), device="cuda")

# Initialize an output tensor
output_tensor = torch.empty_like(input_tensor)

# Compute the softmax
softmax(input_tensor, output_tensor)

# Print the first few elements of the output tensor
print(output_tensor[:5, :5])
