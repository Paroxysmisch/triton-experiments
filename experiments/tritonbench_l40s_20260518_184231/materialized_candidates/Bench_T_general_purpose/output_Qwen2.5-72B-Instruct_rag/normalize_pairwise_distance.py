import torch
import triton
import triton.language as tl

# Triton kernel for computing pairwise distances
@triton.jit
def _pairwise_distance_kernel(
    x1_ptr, x2_ptr, dist_ptr, N, D, x1numel, x2numel, rnumel, p_distance: tl.float32, eps_distance: tl.float32,
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * XBLOCK
    offsets = block_start + tl.arange(0, XBLOCK)
    mask = offsets < x1numel

    row1_indices = D * (offsets // N)
    row2_indices = D * (offsets % N)

    acc = tl.zeros([XBLOCK, RBLOCK], dtype=tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindices = roffset + tl.arange(0, RBLOCK)
        rmask = rindices < rnumel

        mask = rmask & mask
        in1_ptrs = row1_indices + rindices
        in2_ptrs = row2_indices + rindices

        data1 = tl.load(x1_ptr + in1_ptrs, mask, eviction_policy="evict_last", other=0)
        data2 = tl.load(x2_ptr + in2_ptrs, mask, eviction_policy="evict_last", other=0)

        diff = data1 - data2
        diff_pow = tl.pow(tl.abs(diff), p_distance)
        acc += diff_pow

    row_sum = tl.sum(acc, axis=1)
    row_sum = tl.where(mask, row_sum, 0.0)
    row_sum = tl.where(row_sum > eps_distance, row_sum, eps_distance)
    tl.store(dist_ptr + offsets, row_sum, mask)

# Triton kernel for normalizing the distances
@triton.jit
def _normalize_distances_kernel(
    dist_ptr, norm_ptr, out_ptr, N, D, xnumel, p_norm: tl.float32, dim_norm: tl.int32, eps_norm: tl.float32,
    XBLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * XBLOCK
    offsets = block_start + tl.arange(0, XBLOCK)
    mask = offsets < xnumel

    if dim_norm == 1:
        row_indices = offsets // N
        col_indices = offsets % N
        row_norms = tl.zeros([XBLOCK], dtype=tl.float32)
        for i in range(N):
            idx = row_indices * N + i
            row_norms += tl.pow(tl.load(dist_ptr + idx, mask, eviction_policy="evict_last", other=0), p_norm)
        row_norms = tl.pow(row_norms, 1.0 / p_norm)
        row_norms = tl.where(row_norms > eps_norm, row_norms, eps_norm)
        normalized = tl.load(dist_ptr + offsets, mask, eviction_policy="evict_last", other=0) / row_norms
        tl.store(out_ptr + offsets, normalized, mask)
    else:
        col_indices = offsets // N
        row_indices = offsets % N
        col_norms = tl.zeros([XBLOCK], dtype=tl.float32)
        for i in range(N):
            idx = i * N + col_indices
            col_norms += tl.pow(tl.load(dist_ptr + idx, mask, eviction_policy="evict_last", other=0), p_norm)
        col_norms = tl.pow(col_norms, 1.0 / p_norm)
        col_norms = tl.where(col_norms > eps_norm, col_norms, eps_norm)
        normalized = tl.load(dist_ptr + offsets, mask, eviction_policy="evict_last", other=0) / col_norms
        tl.store(out_ptr + offsets, normalized, mask)

# Wrapper function
def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    N, D = x1.shape

    x1numel = N * N
    x2numel = D
    rnumel = D

    # Compute pairwise distances
    dist = torch.empty_strided((N, N), (N, 1), device="cuda", dtype=torch.float32)
    XBLOCK = min(triton.next_power_of_2(N), 1024)
    RBLOCK = min(triton.next_power_of_2(D), 1024)
    g = (1 + (x1numel - 1) // XBLOCK, 1, 1)
    _pairwise_distance_kernel[g](
        x1, x2, dist, N, D, x1numel, x2numel, rnumel, p_distance, eps_distance, XBLOCK=XBLOCK, RBLOCK=RBLOCK
    )

    # Normalize distances
    norm = torch.empty_strided((N, N), (N, 1), device="cuda", dtype=torch.float32)
    out = torch.empty_strided((N, N), (N, 1), device="cuda", dtype=torch.float32)
    XBLOCK = min(triton.next_power_of_2(N * N), 1024)
    g = (1 + (N * N - 1) // XBLOCK, 1, 1)
    _normalize_distances_kernel[g](
        dist, norm, out, N, D, N * N, p_norm, dim_norm, eps_norm, XBLOCK=XBLOCK
    )

    if not keepdim:
        out = out.squeeze(dim_norm)

    return out

# Example usage
x1 = torch.randn(10, 5, device="cuda")
x2 = torch.randn(10, 5, device="cuda")
result = normalize_pairwise_distance(x1, x2)
print(result)
