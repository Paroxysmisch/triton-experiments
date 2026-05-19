import triton
import triton.language as tl

@triton.jit
def digamma_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Block indices
    pid = tl.program_id(0)
    # Block offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to prevent out-of-bounds memory access
    mask = offsets < n_elements

    # Load data from global memory
    x = tl.load(input_ptr + offsets, mask=mask)

    # Compute digamma function
    result = tl.math.digamma(x)

    # Handle the special case for x = 0
    result = tl.where(x == 0, float('-inf'), result)

    # Store result back to global memory
    tl.store(output_ptr + offsets, result, mask=mask)

import torch

def digamma(input, *, out=None):
    # Ensure input is a contiguous tensor
    input = input.contiguous()

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Number of elements
    n_elements = input.numel()

    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 1024

    # Launch the Triton kernel
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    digamma_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out
