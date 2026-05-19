import torch
import triton
import triton.language as tl

@triton.jit
def _triton_normalized_cosine_similarity(x1, x2, dim, eps_similarity, p_norm, eps_norm):
    # Normalize x1 and x2 along the specified dimension
    x1_norm = tl.normalize(x1, dim=dim, p=p_norm, eps=eps_norm)
    x2_norm = tl.normalize(x2, dim=dim, p=p_norm, eps=eps_norm)

    # Calculate cosine similarity
    similarity = x1_norm * x2_norm
    similarity = tl.sum(similarity, axis=dim)

    # Ensure similarity is non-negative
    similarity = tl.where(similarity < 0, 0, similarity)

    # Apply a small epsilon to avoid division by zero
    similarity = tl.where(similarity < eps_similarity, eps_similarity, similarity)

    return similarity

def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> Tensor:
    # Ensure x2 can be broadcasted to x1's shape
    assert x2.shape == x1.broadcast_to(x2.shape).shape

    # Call the Triton kernel for cosine similarity calculation
    return _triton_normalized_cosine_similarity(x1, x2, dim, eps_similarity, p_norm, eps_norm)
