import triton
import triton.language as tl

@triton.jit
def kernel_function(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the starting point for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create a tensor of offsets for element indexing
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to ensure we only operate on valid indices
    mask = offsets < n_elements

    # Load the data from x_ptr
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute the sine of the data
    y = tl.math.sin(x)

    # Store the result in output_ptr
    tl.store(output_ptr + offsets, y, mask=mask)

import torch

def call_kernel(x):
    # Convert input tensor to a Triton-compatible format
    x = x.to(device='cuda', dtype=torch.float32)
    n_elements = x.numel()

    # Create an output tensor
    output = torch.empty_like(x)

    # Define the grid configuration function
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the kernel
    kernel_function[grid](x, output, n_elements, BLOCK_SIZE=1024)

    return output
