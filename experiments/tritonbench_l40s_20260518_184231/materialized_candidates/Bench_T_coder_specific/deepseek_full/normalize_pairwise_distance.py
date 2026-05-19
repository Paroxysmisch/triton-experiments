import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def pairwise_distance(x1, x2, p, eps):
    # Calculate the pairwise distance between x1 and x2 using the specified norm
    return tl.sum(tl.abs(x1 - x2) ** p, axis=-1) ** (1.0 / p) + eps

@triton.jit
def lp_norm(x, p, dim, eps):
    # Compute the lp norm of x along the specified dimension
    return tl.max(tl.sum(tl.abs(x) ** p, axis=dim) ** (1.0 / p), eps)

@triton.jit
def normalize_pairwise_distance(x1, x2, p_distance, eps_distance, keepdim, p_norm, dim_norm, eps_norm):
    # Compute the pairwise distance and normalize it
    distance = pairwise_distance(x1, x2, p_distance, eps_distance)
    norm = lp_norm(distance[:, None], p_norm, dim_norm, eps_norm)
    return distance / norm

def wrapper_normalize_pairwise_distance(x1: Tensor, x2: Tensor, *, p_distance: float = 2.0, eps_distance: float = 1e-6, keepdim: bool = False, p_norm: float = 2, dim_norm: int = 1, eps_norm: float = 1e-12) -> Tensor:
    # Wrapper function for normalize_pairwise_distance
    assert x1.shape == x2.shape
    return normalize_pairwise_distance(x1, x2, p_distance, eps_distance, keepdim, p_norm, dim_norm, eps_norm)
