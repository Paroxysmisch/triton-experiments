import torch
import triton
import triton.language as tl

# Load the Triton kernel
scaled_add_dot_kernel = triton.compile(scaled_add_dot_kernel, signature="(),(),i,f->()", num_warps=4)

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert y.shape == x.shape, "Shapes of y and x must match"
    assert len(y.shape) == 1, "Both y and x must be 1D tensors"
    
    # Ensure y is a contiguous tensor
    y = y.contiguous()
    
    # Launch the Triton kernel
    scaled_add_dot_kernel[(y.numel() // 1024 + 1), 1](y, x, y.numel().item(), alpha)
    
    # Compute the dot product of the modified y with itself
    dot_product = torch.dot(y, y)
    
    return dot_product
