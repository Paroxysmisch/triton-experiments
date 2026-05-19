import torch
import triton
import triton.language as tl

@triton.jit
def masked_add_kernel(
    grad_ptr,
    p_ptr,
    p_mask_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load mask values and convert to boolean
    p_mask_vals = tl.load(p_mask_ptr + offsets, mask=mask, other=0)
    p_mask = p_mask_vals != 0  # Convert to boolean mask

    # Load gradient and parameter values
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0)
    p = tl.load(p_ptr + offsets, mask=mask, other=0)

    # Compute masked addition
    grad += alpha * p * p_mask  # Boolean mask becomes 0/1 in calculation

    # Store result back to gradient tensor
    tl.store(grad_ptr + offsets, grad, mask=mask)

def masked_add(
    grad: torch.Tensor,
    p: torch.Tensor,
    p_mask: torch.Tensor,
    alpha: float
):
    # Validate inputs
    assert all(t.is_cuda for t in (grad, p, p_mask)), "Tensors must be on CUDA"
    assert grad.shape == p.shape == p_mask.shape, "Shape mismatch"
    assert grad.stride() == p.stride() == p_mask.stride(), "Stride mismatch"
    assert all(t.is_contiguous() for t in (grad, p, p_mask)), "Tensors must be contiguous"

    n_elements = grad.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel with optimal block size
    masked_add_kernel[grid](grad, p, p_mask, alpha, n_elements, BLOCK_SIZE=1024)
