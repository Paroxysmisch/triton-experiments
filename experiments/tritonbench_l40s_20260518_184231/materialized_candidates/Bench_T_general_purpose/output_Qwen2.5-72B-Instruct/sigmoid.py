import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    X,  # Pointer to the input tensor
    Y,  # Pointer to the output tensor
    N,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    # Compute the starting index for the current block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create a mask to handle the case where the block size is not a multiple of the tensor size
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load the input data
    x = tl.load(X + offsets, mask=mask)

    # Compute the sigmoid
    y = 1 / (1 + tl.exp(-x))

    # Store the result
    tl.store(Y + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

def sigmoid(input, *, out=None):
    # Ensure the input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)

    # Ensure the output tensor has the same shape and dtype as the input
    if out.shape != input.shape or out.dtype != input.dtype:
        raise ValueError("Output tensor must have the same shape and dtype as the input tensor")

    # Launch the Triton kernel
    grid = (triton.cdiv(input.numel(), 1024),)
    sigmoid_kernel[grid](
        input.contiguous().data_ptr(),
        out.contiguous().data_ptr(),
        input.numel(),
        BLOCK_SIZE=1024,
    )

    return out
