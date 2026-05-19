import torch
import triton
import triton.language as tl

@triton.jit
def _scaled_add_kernel(y_ptr, x_ptr, alpha, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    y_val = tl.load(y_ptr + offsets, mask=mask)
    x_val = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, y_val + alpha * x_val, mask=mask)

def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert y.is_cuda and x.is_cuda, "Tensors must be on GPU."
    assert y.numel() == x.numel(), "Tensors must have the same number of elements."
    n = y.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _scaled_add_kernel[grid](y, x, alpha, n, BLOCK_SIZE=BLOCK_SIZE)
    return torch.norm(y, p=2)
