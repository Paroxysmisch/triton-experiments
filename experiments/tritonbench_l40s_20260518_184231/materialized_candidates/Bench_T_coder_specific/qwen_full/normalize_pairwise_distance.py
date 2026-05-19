import torch
import triton
import triton.language as tl

@triton.jit
def _triton_normalize_pairwise_distance(x1, x2, p_distance, eps_distance, keepdim, p_norm, dim_norm, eps_norm):
    dist = tl.pairwise_distance(x1, x2, p_distance, eps_distance, keepdim)
    norm = tl.norm(dist, p_norm, dim_norm, eps_norm)
    return dist / norm

def normalize_pairwise_distance(x1: torch.Tensor, x2: torch.Tensor, p_distance: float = 2.0, eps_distance: float = 1e-06, keepdim: bool = False, p_norm: int = 2, dim_norm: int = 1, eps_norm: float = 1e-12) -> torch.Tensor:
    """
    For each pair of entities in `x1` and `x2`, this function computes the normalized distance.
    Args:
        x1 (Tensor): the first input tensor
        x2 (Tensor): the second input tensor, must have the same shape as `x1`
        p_distance (float): p-norm used for distance computation. (default: 2.0)
        eps_distance (float): small value to avoid division by zero in distance computation. (default: 1e-6)
        keepdim (bool): whether the reduced dimensions are kept in the output. (default: False)
        p_norm (int): p-norm used for normalization. (default: 2)
        dim_norm (int): dimension used for normalization
        eps_norm (float): small value to avoid division by zero in normalization. (default: 1e-12)
    Returns:
        Tensor: normalized distance
    """
    return _triton_normalize_pairwise_distance(x1, x2, p_distance, eps_distance, keepdim, p_norm, dim_norm, eps_norm)
