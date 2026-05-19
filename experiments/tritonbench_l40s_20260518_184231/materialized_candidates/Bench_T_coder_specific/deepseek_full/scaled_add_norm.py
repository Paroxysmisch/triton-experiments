import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_norm(y: tl.tensor, x: tl.tensor, alpha: tl.tensor) -> tl.tensor:
    # Compute y += alpha * x
    idx = tl.arange(0, 1)
    y = tl.store(y + alpha * tl.load(x + idx), y + idx)
    # Compute 2-norm of the modified y
    norm = tl.sqrt(tl.sum(tl.math.pow(y, 2.0)))
    return norm

def scaled_add_norm_wrapper(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Convert alpha to tensor for triton function
    alpha = torch.tensor(alpha)
    # Call triton function
    return scaled_add_norm(y, x, alpha)
