import torch
import triton
import triton.language as tl

@triton.jit
def masked_add_kernel(
    grad_ptr,
    p_ptr,
    p_mask_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    alpha: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask, other=1).to(tl.int1)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    p = tl.load(p_ptr + offsets, mask=mask)
    result = grad + p * alpha * p_mask
    tl.store(grad_ptr + offsets, result, mask=mask)

def masked_add(grad, p, p_mask, alpha=1.0):
    assert grad.is_cuda and p.is_cuda and p_mask.is_cuda
    assert grad.layout == p.layout == p_mask.layout
    assert grad.stride(0) == p.stride(0) == p_mask.stride(0)
    n_elements = grad.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    masked_add_kernel[grid](grad, p, p_mask, n_elements, BLOCK_SIZE=1024, alpha=alpha)
    return grad
