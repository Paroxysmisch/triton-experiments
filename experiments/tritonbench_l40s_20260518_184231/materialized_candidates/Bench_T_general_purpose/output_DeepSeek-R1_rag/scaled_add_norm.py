import triton
import triton.language as tl
import torch

@triton.jit
def scaled_add_kernel(
    y_ptr, x_ptr, alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    y_updated = y + alpha * x
    tl.store(y_ptr + offsets, y_updated, mask=mask)

def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert y.dim() == 1 and x.dim() == 1, "Tensors must be 1-dimensional"
    assert y.size(0) == x.size(0), "Tensors must have the same length"
    assert y.is_cuda and x.is_cuda, "Tensors must be on GPU"
    n_elements = y.size(0)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    scaled_add_kernel[grid](y, x, alpha, n_elements, BLOCK_SIZE=1024)
    return torch.norm(y, p=2)
