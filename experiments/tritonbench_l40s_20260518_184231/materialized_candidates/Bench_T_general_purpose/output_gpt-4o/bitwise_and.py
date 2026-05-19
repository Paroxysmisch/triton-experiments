import triton
import triton.language as tl

@triton.jit
def bitwise_and_kernel(
    input_ptr, other_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Define the range of indices this program instance will process
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask)
    other_data = tl.load(other_ptr + offsets, mask=mask)

    # Compute bitwise AND
    result = input_data & other_data

    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

import torch

def bitwise_and(input, other, *, out=None):
    assert input.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool), \
        "Input tensor must be of integral or Boolean types"
    assert other.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool), \
        "Other tensor must be of integral or Boolean types"
    assert input.shape == other.shape, "Input and other must have the same shape"

    # Determine the number of elements
    n_elements = input.numel()

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bitwise_and_kernel[grid](input, other, out, n_elements, BLOCK_SIZE=1024)

    return out
