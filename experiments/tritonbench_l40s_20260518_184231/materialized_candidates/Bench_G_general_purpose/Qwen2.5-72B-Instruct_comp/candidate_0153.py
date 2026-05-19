import triton
import triton.language as tl

@triton.jit
def sin_kernel(
    in_ptr0,  # Pointer to the input data
    out_ptr,  # Pointer to the output data
    n_elements,  # Total number of elements to process
    BLOCK_SIZE: tl.constexpr  # Block size (compile-time constant)
):
    # Calculate the starting index for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create a range of offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input data with masking to prevent out-of-bounds access
    mask = offsets < n_elements
    input_data = tl.load(in_ptr0 + offsets, mask=mask)

    # Compute the sine of each element
    output_data = tl.sin(input_data)

    # Store the result back into memory with masking
    tl.store(out_ptr + offsets, output_data, mask=mask)

import torch
import triton
import triton.language as tl

def sin_triton(x: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    assert x.is_cuda, "Input tensor must be on the GPU"

    # Determine the number of elements in the input tensor
    n_elements = x.numel()

    # Allocate memory for the output tensor
    y = torch.empty_like(x)

    # Define the grid size and block size
    grid = (triton.cdiv(n_elements, 4),)
    BLOCK_SIZE = 4

    # Invoke the Triton kernel
    sin_kernel[grid](x, y, n_elements, BLOCK_SIZE)

    return y
