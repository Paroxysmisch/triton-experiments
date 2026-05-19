import triton
import triton.language as tl
import torch
import math

# Kernel function to compute the square root of elements in a tensor
@triton.jit
def sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'input_ptr' with boundary mask
    input_value = tl.load(input_ptr + offset, mask=mask)
    # Compute the square root of the loaded elements
    # Using tl.sqrt will result in NaN for negative inputs as expected
    output_value = tl.sqrt(input_value.to(tl.float32))
    # Store the result in output tensor 'output_ptr' with boundary mask
    tl.store(output_ptr + offset, output_value, mask=mask)

# Function to launch the Triton kernel and compute square root
def sqrt(input, *, out=None):
    # If 'out' is not provided, create an output tensor with the same shape as 'input'
    if out is None:
        out = torch.empty_like(input)
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    sqrt_kernel[(grid_size,)](input, out, n_elements, block_size)
    return out

# Example usage
A = torch.tensor([4.0, 9.0, -1.0, 16.0], dtype=torch.float32)
B = sqrt(A)
print(B)  # Should print tensor([2.0, 3.0, NaN, 4.0])
