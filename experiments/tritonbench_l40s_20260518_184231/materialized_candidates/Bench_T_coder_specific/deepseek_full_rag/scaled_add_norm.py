import triton
import torch

# Triton kernel to perform scaled addition and calculate 2-norm
@triton.jit
def scaled_add_norm_kernel(y_ptr, x_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask)
    tl.store(y_ptr + offset, x + alpha, mask=mask)

# Wrapper function to call the Triton kernel
def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    n_elements = y.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    scaled_add_norm_kernel[grid](y, x, alpha, n_elements, BLOCK_SIZE)
    return torch.norm(y, p=2)

# Example usage
y = torch.randn(1024, device='cuda')
x = torch.randn(1024, device='cuda')
alpha = 2.0
norm = scaled_add_norm(y, x, alpha)
