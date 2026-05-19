import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Digamma Kernel
# This is the kernel function for calculating the digamma function of elements in the input tensor.
# The function uses libdevice.digamma to perform the computation.

@triton.jit
def digamma_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # Get the program ID for the current block
    block_start = pid * BLOCK_SIZE  # Calculate the starting index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Generate offsets for elements in the block
    mask = offsets < n_elements  # Ensure we don't go out of bounds
    x = tl.load(x_ptr + offsets, mask=mask)  # Load input tensor values
    y = libdevice.digamma(x)  # Apply the digamma function from libdevice
    tl.store(y_ptr + offsets, y, mask=mask)  # Store the result in the output tensor

# Wrapper function for the digamma kernel
def digamma(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)  # Create an output tensor if not provided
    n_elements = input.numel()  # Get the number of elements in the tensor
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )  # Define the grid dimensions for Triton
    digamma_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)  # Launch the kernel
    return out

# Example usage
torch.manual_seed(0)
size = 98432  # Size of the input tensor
x = torch.rand(size, device='cuda')  # Create a random input tensor
output_triton = digamma(x)  # Compute the digamma function using Triton
output_torch = torch.special.digamma(x)  # Compute the digamma function using PyTorch for comparison

# Print the results for comparison
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
