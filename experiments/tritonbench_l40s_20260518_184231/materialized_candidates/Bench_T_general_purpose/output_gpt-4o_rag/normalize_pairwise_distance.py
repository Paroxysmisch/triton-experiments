import torch
import triton
import triton.language as tl

@triton.jit
def pairwise_distance_kernel(x1_ptr, x2_ptr, dist_ptr, N, D, p_distance, eps_distance, XBLOCK: tl.constexpr):
    x1_offset = tl.program_id(0) * XBLOCK
    x2_offset = tl.program_id(1) * XBLOCK
    x1_idx = x1_offset + tl.arange(0, XBLOCK)
    x2_idx = x2_offset + tl.arange(0, XBLOCK)

    x1_mask = x1_idx < N
    x2_mask = x2_idx < N

    acc = tl.zeros([XBLOCK, XBLOCK], dtype=tl.float32)

    for d in range(0, D):
        x1_val = tl.load(x1_ptr + x1_idx * D + d, mask=x1_mask, other=0.0)
        x2_val = tl.load(x2_ptr + x2_idx * D + d, mask=x2_mask, other=0.0)
        diff = tl.abs(x1_val - x2_val)
        acc += diff ** p_distance

    dist = (acc + eps_distance) ** (1.0 / p_distance)
    tl.store(dist_ptr + x1_idx * N + x2_idx, dist, mask=x1_mask & x2_mask)

@triton.jit
def normalize_kernel(dist_ptr, norm_ptr, N, dim_norm, p_norm, eps_norm, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    idx = offset + tl.arange(0, XBLOCK)
    mask = idx < N

    acc = tl.zeros([XBLOCK], dtype=tl.float32)
    for n in range(0, N):
        val = tl.load(dist_ptr + idx * N + n, mask=mask, other=0.0)
        acc += val ** p_norm

    norm = (acc + eps_norm) ** (1.0 / p_norm)
    tl.store(norm_ptr + idx, norm, mask=mask)

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    N, D = x1.shape

    dist = torch.empty((N, N), device=x1.device, dtype=torch.float32)
    norm = torch.empty((N,), device=x1.device, dtype=torch.float32)

    XBLOCK = 128  # Example block size

    pairwise_distance_kernel[(N // XBLOCK, N // XBLOCK)](
        x1, x2, dist, N, D, p_distance, eps_distance, XBLOCK=XBLOCK
    )

    normalize_kernel[(N // XBLOCK,)](
        dist, norm, N, dim_norm, p_norm, eps_norm, XBLOCK=XBLOCK
    )

    normalized_dist = dist / norm.unsqueeze(dim_norm)
    if not keepdim:
        normalized_dist = normalized_dist.squeeze(dim_norm)

    return normalized_dist

# Example usage
x1 = torch.rand((256, 128), device='cuda')
x2 = torch.rand((256, 128), device='cuda')
result = normalize_pairwise_distance(x1, x2)
