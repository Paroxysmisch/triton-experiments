import torch
import triton

@triton.jit
def _tensordot_rsqrt(a, b, dims):
    # tensordot a and b over specified dimensions
    output = torch.tensordot(a, b, dims=dims)
    # apply the reciprocal of the square root to the result
    output = torch.rsqrt(output)
    return output

def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    return _tensordot_rsqrt(a, b, dims)
