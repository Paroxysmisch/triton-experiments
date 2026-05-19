import torch
import triton
import triton.language as tl

# Define constants
BLOCK_SIZE = 32

@triton.autotune(
    configs=[
        triton.Config({'block_size': BLOCK_SIZE}, num_stages=1, num_warps=4),
    ],
    key=['n_rows', 'n_cols']
)
def normalize_pairwise_distance_forward_kernel(
    x1_ptr, x2_ptr, dist_ptr, norm_ptr,
    n_rows, n_cols, stride_x1, stride_x2, stride_dist, stride_norm,
    p_distance, eps_distance, p_norm, eps_norm,
    block_size: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(n_rows * n_cols, block_size)
    
    coords = pid * block_size + tl.arange(0, block_size)
    row_idx = coords // n_cols
    col_idx = coords % n_cols
    
    if row_idx >= n_rows or col_idx >= n_cols:
        return
    
    x1 = tl.load(x1_ptr + row_idx * stride_x1 + col_idx * stride_x2)
    x2 = tl.load(x2_ptr + row_idx * stride_x1 + col_idx * stride_x2)
    diff = x1 - x2
    squared_diff = diff * diff
    distance = tl.sum(squared_diff, axis=0)
    distance = tl.sqrt(distance + eps_distance)
    tl.store(dist_ptr + row_idx * stride_dist + col_idx, distance)
    
    distance = tl.load(dist_ptr + row_idx * stride_dist + col_idx)
    max_distance = tl.max(distance)
    norm_value = tl.pow(max_distance, p_norm)
    norm_value = tl.where(norm_value == 0, eps_norm, norm_value)
    normalized_distance = distance / norm_value
    tl.store(norm_ptr + row_idx * stride_norm + col_idx, normalized_distance)


def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    assert x1.shape == x2.shape, "Input tensors must have the same shape"
    
    n_rows, n_cols = x1.shape
    
    # Allocate memory for distances and norms
    dist = torch.empty_like(x1, device=x1.device)
    norm = torch.empty_like(x1, device=x1.device)
    
    # Compute pairwise distances
    pairwise_distance_forward_kernel[grid_size=(n_rows * n_cols, 1)](
        x1.contiguous().data_ptr(), x2.contiguous().data_ptr(), dist.data_ptr(),
        n_rows, n_cols, x1.stride(0), x2.stride(0), dist.stride(0),
        p_distance, eps_distance, p_norm, eps_norm, BLOCK_SIZE
    )
    
    # Normalize distances
    normalize_kernel[grid_size=(n_rows * n_cols, 1)](
        dist.data_ptr(), norm.data_ptr(),
        n_rows, n_cols, dist.stride(0), norm.stride(0),
        p_norm, eps_norm, BLOCK_SIZE
    )
    
    if not keepdim:
        norm = norm.squeeze(dim_norm)
    
    return norm
