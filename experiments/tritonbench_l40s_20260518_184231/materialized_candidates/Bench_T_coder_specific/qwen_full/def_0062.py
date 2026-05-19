import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_dot_kernel(y, x, alpha):
    idx = tl.arange(0, 1024)
    y_ptrs = y + idx
    x_ptrs = x + idx
    mask = idx < y.shape[0]
    y_val = tl.load(y_ptrs, mask=mask)
    x_val = tl.load(x_ptrs, mask=mask)
    scaled_x = x_val * alpha
    y_new = y_val + scaled_x
    tl.store(y_ptrs, y_new, mask=mask)

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert y.is_cuda and x.is_cuda
    n = y.shape[0]
    grid = lambda meta: (triton.cdiv(n, 1024),)
    scaled_add_dot_kernel[grid](y, x, alpha)
    return torch.dot(y, y)
