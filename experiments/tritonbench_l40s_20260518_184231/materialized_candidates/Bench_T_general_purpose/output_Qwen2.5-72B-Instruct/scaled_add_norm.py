import triton
import triton.language as tl

@triton.jit
def scaled_add_kernel(
    y_ptr,  # Pointer to the y tensor
    x_ptr,  # Pointer to the x tensor
    alpha,  # Scalar multiplier for x
    n,      # Number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    y = tl.load(y_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y += alpha * x
    tl.store(y_ptr + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

# Triton kernel configuration
BLOCK_SIZE = 1024

def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Ensure the tensors are on the same device and have the same shape
    assert y.device == x.device, "Tensors must be on the same device"
    assert y.shape == x.shape, "Tensors must have the same shape"
    
    n = y.numel()
    
    # Grid and block configuration
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch the Triton kernel
    scaled_add_kernel[grid](y, x, alpha, n, BLOCK_SIZE)
    
    # Compute the 2-norm of the updated y
    norm = torch.norm(y, p=2)
    
    return norm
