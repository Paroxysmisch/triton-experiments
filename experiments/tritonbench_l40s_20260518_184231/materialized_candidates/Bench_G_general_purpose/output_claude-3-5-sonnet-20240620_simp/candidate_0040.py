import torch
import triton
import triton.language as tl

@triton.jit
def masked_add_kernel(
    grad_ptr,        # Pointer to gradient tensor
    p_data_ptr,      # Pointer to parameter data tensor
    mask_ptr,        # Pointer to mask tensor
    alpha,           # Scaling factor
    n_elements,      # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    # Calculate the absolute position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements
    mask = offsets < n_elements
    
    # Load data with boundary checks
    grad = tl.load(grad_ptr + offsets, mask=mask)
    p_data = tl.load(p_data_ptr + offsets, mask=mask)
    mask_values = tl.load(mask_ptr + offsets, mask=mask)
    
    # Perform masked addition: grad += alpha * p_data where mask == 0
    mask_condition = mask_values == 0
    grad = tl.where(mask_condition, grad + alpha * p_data, grad)
    
    # Store results
    tl.store(grad_ptr + offsets, grad, mask=mask)

def masked_add(grad, p_data, mask, alpha=1.0):
    """
    Wrapper function for masked_add_kernel
    
    Args:
        grad: Tensor to be modified
        p_data: Tensor to be added
        mask: Binary mask tensor (0 where addition should occur)
        alpha: Scaling factor for p_data
    """
    # Ensure inputs are contiguous
    grad = grad.contiguous()
    p_data = p_data.contiguous()
    mask = mask.contiguous()
    
    # Get total number of elements
    n_elements = grad.numel()
    
    # Define block size (can be tuned for better performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    masked_add_kernel[grid](
        grad_ptr=grad,
        p_data_ptr=p_data,
        mask_ptr=mask,
        alpha=alpha,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return grad
