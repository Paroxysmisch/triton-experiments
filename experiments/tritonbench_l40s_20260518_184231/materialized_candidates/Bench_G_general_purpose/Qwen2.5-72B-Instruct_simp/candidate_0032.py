import triton
import triton.language as tl

@triton.jit
def _dropout(x_ptr, x_keep_ptr, output_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr):
    # Compute the block ID and the starting index for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the offsets for the elements in this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the input and mask data
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
    x_keep = tl.load(x_keep_ptr + offsets, mask=offsets < n_elements)

    # Apply the dropout operation
    scale = 1.0 / (1.0 - p)
    output = tl.where(x_keep != 0, x * scale, 0.0)

    # Store the result
    tl.store(output_ptr + offsets, output, mask=offsets < n_elements)

import torch
import triton
import triton.language as tl

def dropout(x, p):
    # Ensure the input tensor is on the GPU
    assert x.is_cuda, "Input tensor must be on the GPU"

    # Generate the dropout mask
    x_keep = torch.rand_like(x) > p

    # Convert the mask to a float tensor
    x_keep = x_keep.to(dtype=x.dtype)

    # Define the grid size
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Allocate output tensor
    output = torch.empty_like(x)

    # Launch the Triton kernel
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE)

    return output
