import triton
import triton.language as tl

@triton.jit
def relu_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.where(x > 0, x, 0)
    tl.store(Y + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

def relu(input, inplace=False):
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    # Determine the grid and block sizes
    grid = (triton.cdiv(input.numel(), 1024),)
    block = (1024,)

    # Launch the kernel
    relu_kernel[grid, block](input, output, input.numel(), BLOCK_SIZE=1024)

    return output
