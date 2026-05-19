import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_global_transpose(
    A, B,
    stride_am, stride_an,
    stride_bm, stride_bn,
    M, N,
    absmax_inv,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_groups_m = (num_blocks_m + GROUP_M - 1) // GROUP_M
    
    group_start = pid_m * GROUP_M
    group_end = tl.minimum(group_start + GROUP_M, num_blocks_m)
    
    for block_m in range(group_start, group_end):
        a_row_start = block_m * BLOCK_M
        a_rows = a_row_start + tl.arange(0, BLOCK_M)
        a_col_start = pid_n * BLOCK_N
        a_cols = a_col_start + tl.arange(0, BLOCK_N)
        
        mask_a = (a_rows < M)[:, None] & (a_cols < N)[None, :]
        a_ptrs = A + a_rows[:, None] * stride_am + a_cols[None, :] * stride_an
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        quantized = (a * absmax_inv * 127.0).to(tl.int8)
        
        b_row_start = a_col_start
        b_col_start = a_row_start
        
        b_rows = b_row_start + tl.arange(0, BLOCK_N)
        b_cols = b_col_start + tl.arange(0, BLOCK_M)
        
        mask_b = (b_rows < N)[:, None] & (b_cols < M)[None, :]
        transposed_quantized = tl.trans(quantized)
        b_ptrs = B + b_rows[:, None] * stride_bm + b_cols[None, :] * stride_bn
        tl.store(b_ptrs, transposed_quantized, mask=mask_b)

def quantize_global_transpose(A: torch.Tensor):
    M, N = A.shape
    device = A.device
    assert A.is_contiguous(), "Input tensor must be contiguous"
    
    absmax = torch.max(torch.abs(A)).item()
    eps = 1e-5
    absmax_inv = 1.0 / (absmax + eps)
    
    B = torch.empty((N, M), dtype=torch.int8, device=device)
    
    BLOCK_M = 128
    BLOCK_N = 128
    GROUP_M = 8
    
    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_groups_m = (num_blocks_m + GROUP_M - 1) // GROUP_M
    grid_m = num_groups_m
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)
    
    _quantize_global_transpose[grid](
        A, B,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        M, N,
        absmax_inv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_M=GROUP_M,
    )
    return B
