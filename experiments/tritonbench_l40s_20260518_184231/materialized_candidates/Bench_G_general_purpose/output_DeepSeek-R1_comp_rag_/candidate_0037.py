import torch
import triton
import triton.language as tl

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
    
    # Load mask and combine with bounds check
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    
    # Load data elements using combined mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    
    # Compute and store the updated gradient
    grad += p * alpha
    tl.store(grad_ptr + offsets, grad, mask=mask)

def masked_add_(
    grad: torch.Tensor,
    p_data: torch.Tensor,
    p_mask: torch.Tensor,
    alpha: float = 0.0
):
    assert all(t.is_cuda for t in (grad, p_data, p_mask)), "All tensors must be on CUDA"
    assert grad.stride() == p_data.stride() == p_mask.stride(), "Strides must match"
    assert grad.shape == p_data.shape == p_mask.shape, "Shapes must match"
    
    n_elements = grad.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the kernel
    masked_add_kernel[grid](grad, p_data, p_mask, n_elements, alpha, BLOCK_SIZE=1024)
