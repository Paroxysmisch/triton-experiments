import triton
import torch
import triton.language as tl

# Triton kernel function
@triton.jit
def scaled_add_kernel(y_ptr, x_ptr, alpha, n_elements, BLOCK_SIZE: int):
    """
    This kernel function performs the operation y += alpha * x and is executed in parallel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    y += alpha * x
    tl.store(y_ptr + offsets, y, mask=mask)

# Wrapper function to launch the Triton kernel
def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes y += alpha * x and returns the 2-norm of the modified y.
    
    Parameters:
    y (Tensor): The target tensor to be modified, of shape (n,).
    x (Tensor): The tensor to be scaled and added to y, of shape (n,).
    alpha (float): The scalar multiplier for x.
    
    Returns:
    Tensor: The 2-norm of the updated y.
    """
    assert y.shape == x.shape, "Tensors y and x must have the same shape."
    n_elements = y.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the Triton kernel
    scaled_add_kernel[grid](y, x, alpha, n_elements, BLOCK_SIZE)
    
    # Calculate and return the 2-norm of the updated y
    norm = torch.norm(y, p=2)
    return norm
