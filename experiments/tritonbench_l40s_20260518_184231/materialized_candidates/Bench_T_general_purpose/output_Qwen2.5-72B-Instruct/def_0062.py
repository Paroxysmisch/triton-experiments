import triton
import triton.language as tl

@triton.jit
def scaled_add_kernel(y_ptr, x_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y += alpha * x
    tl.store(y_ptr + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert y.dim() == 1, "y must be a 1D tensor"
    assert x.dim() == 1, "x must be a 1D tensor"
    assert y.shape == x.shape, "y and x must have the same shape"
    
    n_elements = y.numel()
    y_ptr = y.contiguous().to(device='cuda', dtype=torch.float32).data_ptr()
    x_ptr = x.contiguous().to(device='cuda', dtype=torch.float32).data_ptr()
    
    grid = (triton.cdiv(n_elements, 1024),)
    scaled_add_kernel[grid](y_ptr, x_ptr, alpha, n_elements, BLOCK_SIZE=1024)
    
    dot_product = torch.dot(y, y)
    return dot_product
