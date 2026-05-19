import triton
import triton.language as tl

@triton.jit
def exp_sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the program ID and offset
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load input elements
    input_elements = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute the exponential and square root
    exp_elements = tl.exp(input_elements)
    sqrt_elements = tl.sqrt(exp_elements)

    # Store the result in the output tensor
    tl.store(output_ptr + offsets, sqrt_elements, mask=offsets < n_elements)

import torch

def exp_sqrt(input, out=None):
    # Ensure the input is a contiguous tensor
    input = input.contiguous()
    n_elements = input.numel()

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Determine the number of blocks and block size
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    exp_sqrt_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
