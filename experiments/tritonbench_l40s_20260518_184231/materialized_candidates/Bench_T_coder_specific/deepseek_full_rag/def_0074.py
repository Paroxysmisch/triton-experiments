import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def cosine_similarity_kernel(
    x1_ptr,
    x2_ptr,
    output_ptr,
    n_elements,
    dim,
    eps_similarity: tl.constexpr,
    p_norm: tl.constexpr,
    eps_norm: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x1 = tl.load(x1_ptr + offsets, mask=mask)
    x2 = tl.load(x2_ptr + offsets, mask=mask)

    x1_norm = tl.sum(tl.pow(tl.abs(x1) ** p_norm, p_norm), dim) / (eps_norm + tl.sum(tl.pow(tl.abs(x1) ** p_norm, p_norm), dim))
    x2_norm = tl.sum(tl.pow(tl.abs(x2) ** p_norm, p_norm), dim) / (eps_norm + tl.sum(tl.pow(tl.abs(x2) ** p_norm, p_norm), dim))

    numerator = tl.sum(x1 * tl.trans(x2), dim)
    denominator = x1_norm * tl.trans(x2_norm)

    output = numerator / (denominator + eps_similarity)
    tl.store(output_ptr + offsets, output, mask=mask)


def normalized_cosine_similarity(
    x1: Tensor,
    x2: Tensor,
    dim: int = 1,
    eps_similarity: float = 1e-8,
    p_norm: float = 2,
    eps_norm: float = 1e-12,
) -> Tensor:
    if x2.ndim == 1:
        x2 = x2.unsqueeze(0)
    x2 = x2.expand(x1.size())

    n_elements = x1.numel()
    output = torch.empty_like(x1)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    cosine_similarity_kernel[grid](
        x1,
        x2,
        output,
        n_elements,
        dim,
        eps_similarity,
        p_norm,
        eps_norm,
    )
    return output
