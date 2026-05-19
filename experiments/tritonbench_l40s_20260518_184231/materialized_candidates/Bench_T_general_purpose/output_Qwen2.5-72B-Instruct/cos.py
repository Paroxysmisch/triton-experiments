import triton
import triton.language as tl

@triton.jit
def cos_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Start index for the block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Offsets for the block
    mask = offsets < n_elements  # Mask to handle the last block

    input_block = tl.load(input_ptr + offsets, mask=mask)  # Load input block
    output_block = tl.cos(input_block)  # Compute cosine
    tl.store(output_ptr + offsets, output_block, mask=mask)  # Store output block

import torch
import triton

def cos(input, *, out=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    # Ensure input is on the same device as the output if provided
    if out is not None and out.device != input.device:
        raise ValueError("input and out must be on the same device")

    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Ensure output tensor has the same shape and dtype as input
    if out.shape != input.shape or out.dtype != input.dtype:
        raise ValueError("out must have the same shape and dtype as input")

    # Launch the Triton kernel
    grid = (triton.cdiv(input.numel(), 1024),)
    cos_kernel[grid](
        input.contiguous().data_ptr(),
        out.contiguous().data_ptr(),
        input.numel(),
        BLOCK_SIZE=1024,
    )

    return out
