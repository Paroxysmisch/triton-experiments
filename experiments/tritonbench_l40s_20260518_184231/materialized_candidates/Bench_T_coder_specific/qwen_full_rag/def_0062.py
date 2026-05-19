import triton
import triton.language as tl
import torch

@triton.jit
def _scaled_add_dot(y, x, n_elements, alpha):
    pid = tl.program_id(axis=0)
    offset = pid * n_elements
    mask = offset < y.shape[0]
    x = tl.load(x + offset, mask=mask)
    y_curr = tl.load(y + offset, mask=mask)
    y_new = y_curr + alpha * x
    tl.store(y + offset, y_new, mask=mask)
    return y_new

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert x.shape == y.shape
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    y = _scaled_add_dot[grid](y, x, n_elements, alpha, BLOCK_SIZE=1024)
    return torch.dot(y, y)
