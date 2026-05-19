import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> Tensor:
    # Normalize x1 and x2 along the specified dimension using L_p norm
    x1_norm = tl.norm(x1.to(tl.float32), p=p_norm, dim=dim, keepdim=True)
    x1_norm = tl.maximum(x1_norm, eps_norm)
    x1_normalized = x1 / x1_norm

    x2_norm = tl.norm(x2.to(tl.float32), p=p_norm, dim=dim, keepdim=True)
    x2_norm = tl.maximum(x2_norm, eps_norm)
    x2_normalized = x2 / x2_norm

    # Compute cosine similarity between the normalized tensors
    similarity = tl.sum(x1_normalized * x2_normalized, axis=dim) / (tl.maximum(x1_norm, eps_norm) * tl.maximum(x2_norm, eps_norm))

    # Ensure the similarity is not less than eps_similarity
    similarity = tl.maximum(similarity, eps_similarity)

    return similarity
