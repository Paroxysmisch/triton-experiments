import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def kernel_function(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the starting point for the block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create an offsets tensor for element indexing
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to ensure operations only occur on valid indices
    mask = offsets < n_elements
    
    # Load data from x_ptr
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute the sine of the loaded data
    result = tl.math.sin(x)
    
    # Store the result in output_ptr
    tl.store(output_ptr + offsets, result, mask=mask)

# Define the wrapper function to call the kernel
def call_kernel(x, output, block_size=1024):
    # Calculate the total number of elements
    n_elements = x.shape[0]
    
    # Define the grid configuration function
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the kernel
    kernel_function[grid](x, output, n_elements, BLOCK_SIZE=block_size)

# Example usage
import torch

# Create input and output tensors
x = torch.randn(1024 * 1024, device='cuda')
output = torch.empty_like(x)

# Call the kernel
call_kernel(x, output)

# Print the first few elements of the output
print(output[:10])
