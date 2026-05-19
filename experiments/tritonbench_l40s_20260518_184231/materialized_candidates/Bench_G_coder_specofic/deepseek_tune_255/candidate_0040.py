import torch
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

# Triton kernel for masked addition
@triton.jit
def masked_add_kernel(
    grad_ptr,
    p_ptr,
    p_mask_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask, other=0.0)
    p_mask = tl.int1(p_mask)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    grad = grad + p * alpha
    grad = tl.where(p_mask, grad, 0.0)
    tl.store(grad_ptr + offsets, grad, mask=mask)

# Wrapper function for masked addition
@triton_helpers.do_not_specialize(["alpha"])
@triton_helpers.do_not_specialize(["p_mask_ptr"])
@triton_helpers.do_not_specialize(["p_ptr"])
@triton_helpers.do_not_specialize(["grad_ptr"])
def masked_add(
    grad_ptr: torch.Tensor,
    p_ptr: torch.Tensor,
    p_mask_ptr: torch.Tensor,
    alpha: float,
    n_elements: int,
    BLOCK_SIZE: int = 1024,
):
    assert grad_ptr.is_cuda and p_ptr.is_cuda and p_mask_ptr.is_cuda
    assert grad_ptr.stride() == p_ptr.stride() == p_mask_ptr.stride() == 1
    assert grad_ptr.layout == p_ptr.layout == p_mask_ptr.layout == torch.sparse_coo
    assert grad_ptr.shape[0] == p_ptr.shape[0] == p_mask_ptr.shape[0] == n_elements
    grid(n_elements, BLOCK_SIZE)
    masked_add_kernel[grid](
        grad_ptr,
        p_ptr,
        p_mask_ptr,
        n_elements,
        alpha,
        BLOCK_SIZE,
    )

# Example usage
grad = torch.randn(1000, device="cuda", dtype=torch.float16)
p = torch.randn(1000, device="cuda", dtype=torch.float16)
p_mask = torch.randn(1000, device="cuda", dtype=torch.bool)
masked_add(grad, p, p_mask, 0.1, grad.numel())
