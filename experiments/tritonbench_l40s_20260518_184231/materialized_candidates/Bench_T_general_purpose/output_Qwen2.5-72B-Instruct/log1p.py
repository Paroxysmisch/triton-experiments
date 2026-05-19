import triton
import triton.language as tl

@triton.jit
def log1p_kernel(X, Y, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask)
    y = tl.log(x + 1.0)
    tl.store(Y + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

def log1p(input, *, out=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    # Ensure input is on the same device as the Triton kernel
    device = input.device

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input, device=device)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input tensor")

    # Launch the Triton kernel
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    log1p_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)

    return out
