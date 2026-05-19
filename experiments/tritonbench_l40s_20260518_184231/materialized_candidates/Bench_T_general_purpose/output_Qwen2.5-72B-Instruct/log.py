import triton
import triton.language as tl

@triton.jit
def log_kernel(
    X,  # Pointer to the input tensor
    Y,  # Pointer to the output tensor
    N,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.log(x)
    tl.store(Y + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

def log(input, *, out=None):
    # Ensure the input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input, device=input.device)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input tensor")

    # Launch the Triton kernel
    grid = (triton.cdiv(input.numel(), 1024),)
    log_kernel[grid](
        input,  # Pointer to the input tensor
        out,    # Pointer to the output tensor
        input.numel(),  # Number of elements in the tensor
        BLOCK_SIZE=1024,  # Block size
    )

    return out
