import triton
import triton.language as tl
import torch

@triton.jit
def _scaled_add_kernel(x_ptr, y_ptr, alpha, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, y_vals + alpha * x_vals, mask=mask)

def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    n = y.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _scaled_add_kernel[grid](x, y, alpha, n, BLOCK_SIZE=BLOCK_SIZE)
    return torch.norm(y)
