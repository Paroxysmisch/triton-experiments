import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_dot_kernel(
    y_ptr,  # Pointer to y tensor
    x_ptr,  # Pointer to x tensor
    alpha,  # Scalar multiplier
    n_elements,  # Number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    # Get program ID
    pid = tl.program_id(0)
    # Compute block start/end
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements)
    
    # Load y and x values
    y = tl.load(y_ptr + block_start + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < (block_end - block_start))
    x = tl.load(x_ptr + block_start + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < (block_end - block_start))
    
    # Perform y += alpha * x
    y = y + alpha * x
    
    # Store updated y values
    tl.store(y_ptr + block_start + tl.arange(0, BLOCK_SIZE), y, mask=tl.arange(0, BLOCK_SIZE) < (block_end - block_start))
    
    # Compute partial dot product
    partial_dot = tl.sum(y * y, axis=0)
    
    # Return partial dot product for reduction
    return partial_dot

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes y += alpha * x and returns the dot product of the modified y with itself.
    
    Args:
        y (torch.Tensor): The target tensor to be modified, of shape (n,)
        x (torch.Tensor): The tensor to be scaled and added to y, of shape (n,)
        alpha (float): The scalar multiplier for x
        
    Returns:
        torch.Tensor: The dot product of the modified y with itself
    """
    assert y.is_contiguous(), "Input tensor y must be contiguous"
    assert x.is_contiguous(), "Input tensor x must be contiguous"
    assert y.shape == x.shape, "Input tensors must have the same shape"
    assert y.dtype == x.dtype, "Input tensors must have the same dtype"
    
    n_elements = y.numel()
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Initialize output tensor for dot product
    dot_product = torch.zeros(1, device=y.device, dtype=y.dtype)
    
    # Launch kernel
    partial_dots = scaled_add_dot_kernel[grid](
        y_ptr=y.data_ptr(),
        x_ptr=x.data_ptr(),
        alpha=alpha,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Sum partial dot products
    return partial_dots.sum()
