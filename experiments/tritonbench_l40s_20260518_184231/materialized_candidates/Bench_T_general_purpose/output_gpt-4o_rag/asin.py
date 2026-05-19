import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Asin Kernel
@triton.jit
def asin_kernel(
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
    x = libdevice.asin(x)  # Apply the asin function from libdevice
    tl.store(y_ptr + offsets, x, mask=mask)  # Store the result in the output tensor

# Wrapper function for the asin kernel
def asin(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)  # Create an output tensor if not provided

    assert input.is_cuda and out.is_cuda, "Input and output tensors must be on CUDA device"

    n_elements = input.numel()  # Get the number of elements in the input tensor
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)  # Define the grid dimensions for Triton

    # Launch the Triton kernel
    asin_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)

    return out

# Example usage
torch.manual_seed(0)
size = 98432  # Size of the input tensor
x = torch.rand(size, device='cuda') * 2 - 1  # Create a random input tensor in range [-1, 1]
output_triton = asin(x)  # Compute the arc sine using the Triton wrapper
output_torch = torch.asin(x)  # Compute the arc sine using PyTorch

# Print the results for comparison
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
