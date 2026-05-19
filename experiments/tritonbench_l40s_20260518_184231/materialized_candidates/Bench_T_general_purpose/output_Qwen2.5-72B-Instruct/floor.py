import triton
import triton.language as tl

@triton.jit
def floor_kernel(
    input_ptr,  # *Pointer* to the input tensor
    output_ptr,  # *Pointer* to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Compute the start index for this block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Compute the offsets for this block
    mask = offsets < n_elements  # Create a mask to avoid out-of-bounds access

    input_block = tl.load(input_ptr + offsets, mask=mask)  # Load the input block
    output_block = tl.floor(input_block)  # Compute the floor of the input block

    tl.store(output_ptr + offsets, output_block, mask=mask)  # Store the result

import torch
import triton
import triton.language as tl

def floor(input, *, out=None):
    # Check if the input is an integer tensor
    if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        if out is not None:
            out.copy_(input)
        else:
            out = input.clone()
        return out

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)

    # Launch the Triton kernel
    grid = (triton.cdiv(input.numel(), 1024),)
    floor_kernel[grid](
        input.contiguous().data_ptr(),
        out.contiguous().data_ptr(),
        input.numel(),
        BLOCK_SIZE=1024,
    )

    return out
