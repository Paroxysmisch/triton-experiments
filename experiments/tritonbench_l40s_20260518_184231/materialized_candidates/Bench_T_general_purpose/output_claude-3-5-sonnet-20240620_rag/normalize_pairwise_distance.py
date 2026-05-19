import torch
import triton
import triton.language as tl

@triton.jit
def _pairwise_distance(x1_ptr, x2_ptr, dist_ptr, N, D, p_distance: tl.constexpr, eps_distance: tl.constexpr):
    # Compute pairwise distances between x1 and x2
    row_idx = tl.program_id(0) * tl.blockdim_x + tl.arange(0, tl.blockdim_x)
    col_idx = tl.program_id(1) * tl.blockdim_y + tl.arange(0, tl.blockdim_y)

    mask = (row_idx < N) & (col_idx < N)
    dist = tl.zeros((tl.blockdim_x, tl.blockdim_y), dtype=tl.float32)

    for d in range(D):
        x1 = tl.load(x1_ptr + row_idx * D + d, mask)
        x2 = tl.load(x2_ptr + col_idx * D + d, mask)
        dist += (x1 - x2) ** p_distance

    dist = tl.where(mask, dist, 0.0)
    tl.store(dist_ptr + row_idx * N + col_idx, dist, mask)

@triton.jit
def _normalize_distance(dist_ptr, norm_ptr, N, p_norm: tl.constexpr, dim_norm: tl.constexpr, eps_norm: tl.constexpr):
    # Normalize the distances
    row_idx = tl.program_id(0) * tl.blockdim_x + tl.arange(0, tl.blockdim_x)
    mask = row_idx < N

    norm = tl.sqrt(tl.sum(tl.load(dist_ptr + row_idx * N, mask), dim=dim_norm)) + eps_norm
    norm = tl.where(mask, norm, 1.0)  # Avoid division by zero

    dist = tl.load(dist_ptr + row_idx * N, mask)
    normalized_dist = dist / norm
    tl.store(norm_ptr + row_idx * N, normalized_dist, mask)

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    N, D = x1.shape
    dist = torch.empty((N, N), device=x1.device, dtype=torch.float32)

    # Launch pairwise distance kernel
    grid = (triton.cdiv(N, 32), triton.cdiv(N, 32))  # Adjust block size as needed
    _pairwise_distance[grid](x1, x2, dist, N, D, p_distance, eps_distance)

    # Normalize distances
    normalized_dist = torch.empty_like(dist)
    _normalize_distance[grid](dist, normalized_dist, N, p_norm, dim_norm, eps_norm)

    if keepdim:
        return normalized_dist.unsqueeze(dim_norm)
    return normalized_dist
