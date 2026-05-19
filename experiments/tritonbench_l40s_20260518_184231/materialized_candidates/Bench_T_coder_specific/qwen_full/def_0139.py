import torch
import triton
import triton.language as tl
from torch import Tensor
from .utils import calculate_settings

@triton.jit
def triton_std(x, N, correction, keepdim, BLOCK_SIZE: tl.constexpr):
    # Calculate the mean of the block
    mean = tl.sum(x, axis=0) / N
    # Compute the variance
    variance = tl.sum((x - mean) * (x - mean), axis=0) / (N - correction)
    # Return the standard deviation
    return tl.sqrt(variance, axis=0) if keepdim else tl.sqrt(variance)

def std(input: Tensor, dim=None, *, correction=1, keepdim=False, out=None) -> Tensor:
    # Ensure input is a triton tensor
    if not isinstance(input, Tensor):
        input = torch.as_tensor(input)
    # Check if input is empty
    if input.numel() == 0:
        return input
    # Prepare output tensor
    if out is None:
        out = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    else:
        out.copy_(input)
    # Calculate settings for triton kernel
    N, BLOCK_SIZE, num_warps = calculate_settings(input, dim, correction)
    # Call the triton kernel
    triton_std[(1,)](
        input, N, correction, keepdim, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, out=out
    )
    return out
