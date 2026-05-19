import triton
import triton.language as tl
import torch

@triton.jit
def masked_add_kernel(
    grad_ptr,    # Pointer to gradient tensor
    p_ptr,       # Pointer to parameter tensor
    p_mask_ptr,  # Pointer to mask tensor
    n_elements,  # Number of elements
    alpha,       # Scaling factor
    BLOCK_SIZE: tl.constexpr,  # Static block size
):
    # Calculate the program ID and corresponding index
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create an offset array for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for bounds checking
    mask = offsets < n_elements
    
    # Load the mask values and convert to boolean
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask)
    p_mask = tl.int1(p_mask)
    
    # Combine masks for valid elements and masked values
    mask = mask & p_mask
    
    # Load input values
    grad = tl.load(grad_ptr + offsets, mask=mask)
    p = tl.load(p_ptr + offsets, mask=mask)
    
    # Perform masked addition
    output = grad + alpha * p
    
    # Store the result
    tl.store(grad_ptr + offsets, output, mask=mask)

def masked_add(grad, p, p_mask, alpha=1.0):
    """
    Performs masked addition: grad += alpha * p where p_mask is True
    
    Args:
        grad: Input gradient tensor to be modified
        p: Parameter tensor to be added
        p_mask: Boolean mask tensor
        alpha: Scaling factor for p
    """
    assert grad.is_cuda and p.is_cuda and p_mask.is_cuda, "All tensors must be on CUDA"
    assert grad.stride() == p.stride() == p_mask.stride(), "All tensors must have same layout"
    assert grad.shape == p.shape == p_mask.shape, "All tensors must have same shape"
    
    n_elements = grad.numel()
    BLOCK_SIZE = 1024  # Can be tuned for performance
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    masked_add_kernel[grid](
        grad_ptr=grad.data_ptr(),
        p_ptr=p.data_ptr(),
        p_mask_ptr=p_mask.data_ptr(),
        n_elements=n_elements,
        alpha=alpha,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return grad
