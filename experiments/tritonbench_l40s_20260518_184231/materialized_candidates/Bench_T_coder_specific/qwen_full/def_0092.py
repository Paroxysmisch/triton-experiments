import torch
import triton
import triton.language as tl

@triton.jit
def _tensordot_rsqrt_kernel(a, b, dims):
    # Triton kernel to compute the reciprocal of the square root of the tensordot product
    return tl.rsqrt(tl.sum(a.to(tl.float32) * b.to(tl.float32), axis=dims))

def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    """
    Returns the reciprocal of the square root of the tensordot product of a and b.
    """
    return _tensordot_rsqrt_kernel(a, b, dims)
