import torch
import triton
import triton.language as tl

@triton.jit
def fused_pairwise_distance_normalize_kernel(
    x1_ptr, x2_ptr, out_ptr, N, D, ridx_stride, cidx_stride, embed_stride, n_pairs, p_norm, eps_norm, eps_distance, BLOCK_SIZE: tl.constexpr
):
    ridx = tl.program_id(0)
    cidx = tl.program_id(1)

    x1_ptr = x1_ptr + ridx * ridx_stride
    x2_ptr = x2_ptr + cidx * cidx_stride
    out_ptr = out_ptr + (ridx * D + cidx) * embed_stride

    embed_idx = tl.arange(0, BLOCK_SIZE)
    mask = embed_idx < D

    x1 = tl.load(x1_ptr + embed_idx, mask=mask, other=0).to(tl.float32)
    x2 = tl.load(x2_ptr + embed_idx, mask=mask, other=0).to(tl.float32)

    # Normalize x1 and x2
    norm_x1 = tl.math.pow(tl.math.abs(x1), p_norm).sum(axis=0)
    norm_x1 = tl.math.pow(norm_x1 + eps_norm, 1.0 / p_norm)
    x1 = x1 / norm_x1

    norm_x2 = tl.math.pow(tl.math.abs(x2), p_norm).sum(axis=0)
    norm_x2 = tl.math.pow(norm_x2 + eps_norm, 1.0 / p_norm)
    x2 = x2 / norm_x2

    # Compute distance
    distance = x1 - x2
    distance = tl.math.pow(tl.math.abs(distance), p_norm).sum(axis=0)
    distance = tl.math.pow(distance + eps_distance, 1.0 / p_norm)

    # Write distance to output
    tl.store(out_ptr, distance)

def fused_pairwise_distance_normalize(x1: torch.Tensor, x2: torch.Tensor, p_norm: float = 2.0, eps_norm: float = 1e-12, eps_distance: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    assert x1.dim() == 2
    assert x2.dim() == 2
    assert x1.size(1) == x2.size(1)

    D = x1.size(1)
    n_pairs = x1.size(0) * x2.size(0)
    out = torch.empty(n_pairs, dtype=torch.float32, device=x1.device)

    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (x1.size(0), x2.size(0))

    fused_pairwise_distance_normalize_kernel[grid](
        x1, x2, out, D, D, x1.stride(0), x2.stride(0), out.stride(0), n_pairs, p_norm, eps_norm, eps_distance, BLOCK_SIZE
    )

    if not keepdim:
        out = out.view(x1.size(0), x2.size(0))

    return out
