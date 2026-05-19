import triton
import triton.language as tl

@triton.jit
def _dropout(
    x_ptr,  # Pointer to the input tensor
    x_keep_ptr,  # Pointer to the mask tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    p,  # Dropout probability
    BLOCK_SIZE: tl.constexpr  # Block size for parallelization
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Calculate the starting index for the block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Generate offsets for the block
    mask = offsets < n_elements  # Ensure we don't go out of bounds

    x = tl.load(x_ptr + offsets, mask=mask)  # Load elements from the input tensor
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)  # Load elements from the mask tensor

    # Apply dropout: scale retained elements and zero-out others
    output = tl.where(x_keep, x / (1 - p), 0.0)

    tl.store(output_ptr + offsets, output, mask=mask)  # Store the results back to the output tensor

import torch
import triton
import triton.language as tl

def dropout(x: torch.Tensor, x_keep: torch.Tensor, p: float):
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x_keep.is_contiguous(), "Mask tensor must be contiguous"
    assert x.shape == x_keep.shape, "Input and mask tensors must have the same shape"

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE  # Calculate the grid size

    # Allocate memory for the output tensor
    output = torch.empty_like(x)

    # Launch the Triton kernel
    _dropout[grid_size](x, x_keep, output, n_elements, p, BLOCK_SIZE)

    return output
