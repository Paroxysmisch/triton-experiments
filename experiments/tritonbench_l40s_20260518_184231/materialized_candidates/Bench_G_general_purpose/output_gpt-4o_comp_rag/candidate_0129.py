import triton
import triton.language as tl

@triton.jit
def kernel_function(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the starting point of the current block
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets tensor for element indexing
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to ensure operations only occur on valid indices
    mask = offsets < n_elements
    
    # Load data from x_ptr
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute the sine of the input data
    result = tl.math.sin(x)
    # Store the result in output_ptr
    tl.store(output_ptr + offsets, result, mask=mask)

def call_kernel(x_ptr, n_elements, BLOCK_SIZE=1024):
    # Create output tensor
    output = triton.testing.empty_like(x_ptr)
    
    # Define grid configuration
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the kernel
    kernel_function[grid](x_ptr, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

# Example usage
import torch

# Initialize input data
n_elements = 1 << 20  # 1M elements
x = torch.linspace(0, 10, n_elements, device='cuda')

# Call the kernel
output = call_kernel(x, n_elements)

# Verify the result
torch.testing.assert_allclose(output, torch.sin(x))
