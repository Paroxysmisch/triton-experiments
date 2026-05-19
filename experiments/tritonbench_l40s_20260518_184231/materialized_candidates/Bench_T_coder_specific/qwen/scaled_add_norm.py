import torch
from triton import csrc

def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert y.shape == x.shape, "y and x must have the same shape"
    assert len(y.shape) == 1, "y and x must be 1-dimensional tensors"
    
    n = y.size(0)
    grid = (triton.cdiv(n, BLOCK_SIZE), 1)
    
    # Allocate device memory for the inputs and outputs
    y_device = y.contiguous().cuda(non_blocking=True)
    x_device = x.contiguous().cuda(non_blocking=True)
    
    # Launch the Triton kernel
    scaled_add_norm_kernel[grid, BLOCK_SIZE](y_device, x_device, n, alpha)
    
    # Calculate the 2-norm of the updated y
    norm = torch.norm(y_device)
    
    return norm
