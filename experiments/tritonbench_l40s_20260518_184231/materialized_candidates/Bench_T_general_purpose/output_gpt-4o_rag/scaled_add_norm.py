import triton
import triton.language as tl
import torch

# Triton kernel for element-wise operation y += alpha * x
@triton.jit
def scaled_add_kernel(y_ptr, x_ptr, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset, mask=offset < y_ptr.shape[0])
    y = tl.load(y_ptr + offset, mask=offset < y_ptr.shape[0])
    result = y + alpha * x
    tl.store(y_ptr + offset, result, mask=offset < y_ptr.shape[0])

# Wrapper function to launch the Triton kernel and compute the 2-norm
def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert y.shape == x.shape, "y and x must have the same shape"
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(y.numel(), meta['BLOCK_SIZE']),)
    
    # Launch the Triton kernel
    scaled_add_kernel[grid](y, x, alpha, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute and return the 2-norm of the modified y
    norm = torch.norm(y)
    return norm

# Example usage
y = torch.tensor([1.0, 2.0, 3.0], device='cuda')
x = torch.tensor([0.5, 0.5, 0.5], device='cuda')
alpha = 2.0

norm = scaled_add_norm(y, x, alpha)
print("Updated y:", y)
print("2-norm of updated y:", norm)
