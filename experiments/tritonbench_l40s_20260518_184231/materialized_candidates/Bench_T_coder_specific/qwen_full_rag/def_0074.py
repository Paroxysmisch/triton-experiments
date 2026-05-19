import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

@triton.jit
def normalize(x, dim=-1, p=2, eps=1e-12):
    # Calculate the L_p norm of x along the specified dimension
    norm = tl.p_norm(x, p=p, axis=dim, keepdim=True)
    # Avoid division by zero by clamping the norm
    norm = tl.maximum(norm, eps)
    # Normalize x by dividing it by the norm
    return x / norm

@triton.jit
def similarity(x1, x2, eps=1e-8):
    # Compute the dot product of x1 and x2
    x1_dot_x2 = tl.sum(x1 * x2, axis=-1)
    # Avoid division by zero by clamping the dot product
    x1_dot_x2 = tl.maximum(x1_dot_x2, eps)
    return x1_dot_x2

@torch.inference_mode()
def normalized_cosine_similarity(
    x1: Tensor,
    x2: Tensor,
    dim: int = -1,
    eps_similarity: float = 1e-8,
    p_norm: float = 2.0,
    eps_norm: float = 1e-12,
) -> Tensor:
    """Compute the normalized cosine similarity between two tensors.

    Args:
        x1 (Tensor): First input tensor.
        x2 (Tensor): Second input tensor.
        dim (int, optional): Dimension along which to compute the norm. Defaults to -1.
        eps_similarity (float, optional): Epsilon to avoid division by zero in similarity calculation. Defaults to 1e-8.
        p_norm (float, optional): P-norm used for normalization. Defaults to 2.0.
        eps_norm (float, optional): Epsilon to avoid division by zero in norm calculation. Defaults to 1e-12.

    Returns:
        Tensor: Computed similarity scores.
    """
    # Normalize x1 and x2 using the specified p-norm and epsilon
    x1_n = normalize(x1, dim=dim, p=p_norm, eps=eps_norm)
    x2_n = normalize(x2, dim=dim, p=p_norm, eps=eps_norm)
    # Compute the similarity between the normalized tensors
    return similarity(x1_n, x2_n, eps=eps_similarity)
