import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def normalize_kernel(x, output, n_cols, eps, p_norm, BLOCK_N: tl.constexpr):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    # Load input vector
    x_ptr = x + prog_id * n_cols
    v = tl.load(x_ptr + offsets, mask=offsets < n_cols)

    # Compute L_p norm
    norm = tl.norm(v, p=p_norm, dim=0) + eps
    normalized_v = v / norm

    # Store normalized vector
    tl.store(output + prog_id * n_cols + offsets, normalized_v, mask=offsets < n_cols)

@torch.inference_mode()
def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> Tensor:
    """
    Computes the cosine similarity between two normalized input tensors.

    Args:
        x1 (Tensor): First input tensor.
        x2 (Tensor): Second input tensor.
        dim (int): Dimension along which to normalize.
        eps_similarity (float): Small value to avoid division by zero in similarity computation.
        p_norm (float): The order of the norm to use for normalization.
        eps_norm (float): Small value to avoid division by zero in normalization.

    Returns:
        Tensor: The cosine similarity between the normalized tensors.
    """
    
    # Ensure x2 can broadcast to x1's shape
    if x1.shape[dim] != x2.shape[dim]:
        x2 = x2.expand_as(x1)

    n_cols = x1.size(dim)
    seq_len = x1.numel() // n_cols

    # Allocate output tensors for normalized results
    norm_x1 = torch.empty_like(x1)
    norm_x2 = torch.empty_like(x2)

    # Define grid size
    grid = (seq_len,)

    # Normalize x1
    normalize_kernel[grid](x1, norm_x1, n_cols, eps_norm, p_norm, BLOCK_N=triton.next_power_of_2(n_cols))

    # Normalize x2
    normalize_kernel[grid](x2, norm_x2, n_cols, eps_norm, p_norm, BLOCK_N=triton.next_power_of_2(n_cols))

    # Compute cosine similarity
    dot_product = torch.sum(norm_x1 * norm_x2, dim=dim)
    norm_x1_mag = torch.max(torch.norm(norm_x1, p=2, dim=dim), torch.tensor(eps_similarity, device=x1.device))
    norm_x2_mag = torch.max(torch.norm(norm_x2, p=2, dim=dim), torch.tensor(eps_similarity, device=x2.device))

    similarity = dot_product / (norm_x1_mag * norm_x2_mag)

    return similarity
