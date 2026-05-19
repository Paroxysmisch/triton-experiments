import torch
import triton
import triton.language as tl

@triton.jit
def pairwise_distance_kernel(
    x1_ptr, x2_ptr, output_ptr,
    p_distance, eps_distance,
    M, N, K,
    stride_x1m, stride_x1k,
    stride_x2n, stride_x2k,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    off_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    off_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    off_k = tl.arange(0, BLOCK_SIZE_K)
    
    x1_ptrs = x1_ptr + off_m[:, None] * stride_x1m + off_k[None, :] * stride_x1k
    x2_ptrs = x2_ptr + off_n[None, :] * stride_x2n + off_k[:, None] * stride_x2k
    
    mask_m = off_m < M
    mask_n = off_n < N
    mask_k = off_k < K
    
    x1 = tl.load(x1_ptrs, mask=mask_k[None, :] & mask_m[:, None], other=0.0)
    x2 = tl.load(x2_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    
    diff = tl.abs(x1 - x2)
    diff_p = tl.pow(diff, p_distance)
    sum_p = tl.sum(diff_p, axis=1)
    sum_p_eps = sum_p + eps_distance
    distance = tl.pow(sum_p_eps, 1.0 / p_distance)
    
    output_ptrs = output_ptr + off_m[:, None] * stride_om + off_n[None, :] * stride_on
    tl.store(output_ptrs, distance, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def normalize_kernel(
    input_ptr, output_ptr,
    p_norm, eps_norm, dim_norm,
    M, N,
    stride_in_0, stride_in_1,
    stride_out_0, stride_out_1,
    BLOCK_SIZE: tl.constexpr,
):
    if dim_norm == 1:
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < N
        
        row_start = row_idx * stride_in_0
        input_ptrs = input_ptr + row_start + col_offsets * stride_in_1
        row = tl.load(input_ptrs, mask=col_mask, other=0.0)
        
        row_abs = tl.abs(row)
        row_p = tl.pow(row_abs, p_norm)
        norm_p = tl.sum(row_p)
        norm = tl.pow(norm_p, 1.0 / p_norm)
        norm_clamped = tl.maximum(norm, eps_norm)
        
        normalized = row / norm_clamped
        output_ptrs = output_ptr + row_start + col_offsets * stride_out_1
        tl.store(output_ptrs, normalized, mask=col_mask)
    else:
        col_idx = tl.program_id(0)
        row_offsets = tl.arange(0, BLOCK_SIZE)
        row_mask = row_offsets < M
        
        col_start = col_idx * stride_in_1
        input_ptrs = input_ptr + col_start + row_offsets * stride_in_0
        col = tl.load(input_ptrs, mask=row_mask, other=0.0)
        
        col_abs = tl.abs(col)
        col_p = tl.pow(col_abs, p_norm)
        norm_p = tl.sum(col_p)
        norm = tl.pow(norm_p, 1.0 / p_norm)
        norm_clamped = tl.maximum(norm, eps_norm)
        
        normalized = col / norm_clamped
        output_ptrs = output_ptr + col_start + row_offsets * stride_out_0
        tl.store(output_ptrs, normalized, mask=row_mask)

def normalize_pairwise_distance(
    x1: torch.Tensor, x2: torch.Tensor,
    p_distance: float = 2.0, eps_distance: float = 1e-6,
    keepdim: bool = False, p_norm: float = 2,
    dim_norm: int = 1, eps_norm: float = 1e-12
) -> torch.Tensor:
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    M, K = x1.shape
    N, _ = x2.shape
    
    # Compute pairwise distance matrix
    D = torch.empty((M, N), device=x1.device, dtype=x1.dtype)
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)
    
    pairwise_distance_kernel[(grid_m, grid_n)](
        x1, x2, D,
        p_distance, eps_distance,
        M, N, K,
        x1.stride(0), x1.stride(1),
        x2.stride(0), x2.stride(1),
        D.stride(0), D.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Normalize the distance matrix
    if dim_norm not in [0, 1]:
        raise ValueError("dim_norm must be 0 or 1 for 2D tensor")
    
    BLOCK_SIZE = 1024
    if dim_norm == 1:
        grid = (M,)
    else:
        grid = (N,)
    
    normalize_kernel[grid](
        D, D,
        p_norm, eps_norm, dim_norm,
        M, N,
        D.stride(0), D.stride(1),
        D.stride(0), D.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return D
