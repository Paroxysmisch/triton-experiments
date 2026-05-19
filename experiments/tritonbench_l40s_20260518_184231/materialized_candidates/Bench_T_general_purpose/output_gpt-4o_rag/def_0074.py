import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def cosine_similarity_kernel(
    x1_ptr, x2_ptr, output_ptr,
    n_elements, eps_norm, eps_similarity,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID and offset for each block
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    # Load elements from x1 and x2
    x1 = tl.load(x1_ptr + pid * n_elements + offsets, mask=offsets < n_elements, other=0.0)
    x2 = tl.load(x2_ptr + pid * n_elements + offsets, mask=offsets < n_elements, other=0.0)

    # Compute L_p norm for x1 and x2
    norm_x1 = tl.sum(x1**2, axis=0) ** 0.5
    norm_x2 = tl.sum(x2**2, axis=0) ** 0.5

    # Normalize x1 and x2
    x1_normalized = x1 / tl.max(norm_x1, eps_norm)
    x2_normalized = x2 / tl.max(norm_x2, eps_norm)

    # Compute dot product for cosine similarity
    dot_product = tl.sum(x1_normalized * x2_normalized, axis=0)

    # Store the result
    tl.store(output_ptr + pid, dot_product / tl.max(norm_x1 * norm_x2, eps_similarity))

@torch.inference_mode()
def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> Tensor:
    """
    Computes the cosine similarity between two normalized input tensors x1 and x2.

    Args:
        x1 (Tensor): First input tensor.
        x2 (Tensor): Second input tensor.
        dim (int, optional): Dimension along which to compute the similarity. Default is 1.
        eps_similarity (float, optional): Small value to avoid division by zero in similarity computation. Default is 1e-8.
        p_norm (float, optional): Norm degree for normalization. Default is 2.
        eps_norm (float, optional): Small value to avoid division by zero in normalization. Default is 1e-12.

    Returns:
        Tensor: Tensor containing cosine similarities.
    """
    # Ensure x2 can be broadcasted to x1's shape
    if x2.dim() < x1.dim():
        x2 = x2.unsqueeze(dim).expand_as(x1)

    # Get the size of the dimension along which we compute similarity
    n_elements = x1.size(dim)

    # Allocate output tensor
    output = torch.empty(x1.size(0), device=x1.device, dtype=x1.dtype)

    # Define grid size
    grid = (x1.size(0),)

    # Launch Triton kernel
    cosine_similarity_kernel[grid](
        x1, x2, output,
        n_elements, eps_norm, eps_similarity,
        BLOCK_SIZE=triton.next_power_of_2(n_elements),
        num_warps=4,
        num_stages=2
    )

    return output
