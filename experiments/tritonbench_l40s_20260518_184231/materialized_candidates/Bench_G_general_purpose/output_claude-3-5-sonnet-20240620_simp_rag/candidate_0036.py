import torch
import triton
import triton.language as tl

@triton.jit
def masked_add_kernel(
    grad_ptr,    # Pointer to gradient tensor
    p_ptr,       # Pointer to data tensor
    p_mask_ptr,  # Pointer to mask tensor
    n_elements,  # Total number of elements
    alpha,       # Scaling factor
    BLOCK_SIZE: tl.constexpr,  # Size of parallel blocks
):
    # Get the program ID for parallel execution
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load mask values and convert to boolean
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    
    # Combine masks - only process elements where original mask is valid
    # and p_mask is 0 (using ~ to invert p_mask)
    mask = mask & ~p_mask
    
    # Load data values
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    
    # Perform the masked addition: grad += alpha * p
    grad += p * alpha
    
    # Store results back to memory
    tl.store(grad_ptr + offsets, grad, mask=mask)

def masked_add_(grad: torch.Tensor, p_data: torch.Tensor, p_mask: torch.Tensor, alpha: float = 0):
    """
    Wrapper function for masked addition kernel.
    Adds p_data to grad scaled by alpha only where p_mask is 0.
    
    Args:
        grad: Gradient tensor to be modified
        p_data: Data tensor to be added
        p_mask: Mask tensor (0 indicates where addition should occur)
        alpha: Scaling factor for p_data
    """
    # Verify tensors are on CUDA
    assert grad.is_cuda and p_data.is_cuda and p_mask.is_cuda
    
    # Verify tensor layouts are strided
    assert (grad.layout, p_data.layout, p_mask.layout) == (torch.strided, torch.strided, torch.strided)
    
    # Verify matching strides
    assert grad.stride() == p_data.stride() == p_mask.stride()
    
    # Get total number of elements
    n_elements = grad.numel()
    
    # Calculate grid size based on block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    masked_add_kernel[grid](
        grad, p_data, p_mask,
        n_elements, alpha,
        BLOCK_SIZE=1024
    )
