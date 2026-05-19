import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the block ID and the starting index for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create a mask to handle the case where the block size is not a multiple of the number of elements
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input vectors
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform the element-wise addition
    output = x + y

    # Store the result back to the output vector
    tl.store(output_ptr + offsets, output, mask=mask)

import torch

def add(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
    # Ensure the inputs and output are CUDA tensors
    assert x.is_cuda and y.is_cuda and output.is_cuda, "Input and output tensors must be on the same CUDA device."

    # Ensure the input tensors have the same shape
    assert x.shape == y.shape == output.shape, "Input and output tensors must have the same shape."

    # Calculate the number of elements
    n_elements = x.numel()

    # Define the block size
    BLOCK_SIZE = 1024

    # Calculate the grid size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    add_kernel[grid_size, BLOCK_SIZE](
        x, y, output, n_elements, BLOCK_SIZE
    )
