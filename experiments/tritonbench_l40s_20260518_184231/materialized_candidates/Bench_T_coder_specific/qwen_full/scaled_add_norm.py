import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_norm_triton(y, x, alpha):
    # Convert inputs to float32
    y_fp32 = y.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    
    # Compute y += alpha * x
    y_fp32 += alpha * x_fp32
    
    # Compute and return the 2-norm of y
    return torch.norm(y_fp32)

def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Ensure inputs are 1D tensors
    assert y.dim() == 1 and x.dim() == 1
    # Ensure x and y are of the same size
    assert x.size(0) == y.size(0)
    
    # Call the Triton kernel and return the result
    return scaled_add_norm_triton(y, x, alpha)
