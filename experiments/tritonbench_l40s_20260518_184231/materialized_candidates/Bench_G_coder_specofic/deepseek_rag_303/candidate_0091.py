import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128, 
            'BLOCK_SIZE_N': 256, 
            'BLOCK_SIZE_K': 64, 
            'SPLIT_K': 1
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 256, 
            'BLOCK_SIZE_N': 128, 
            'BLOCK_SIZE_K': 64, 
            'SPLIT_K': 1
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 256, 
            'BLOCK_SIZE_N': 64, 
            'BLOCK_SIZE_K': 64, 
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64, 
            'BLOCK_SIZE_N': 256, 
            'BLOCK_SIZE_K': 64, 
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128, 
            'BLOCK_SIZE_N': 128, 
            'BLOCK_SIZE_K': 64, 
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128, 
            'BLOCK_SIZE_N': 64, 
            'BLOCK_SIZE_K': 64, 
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64, 
            'BLOCK_SIZE_N': 128, 
            'BLOCK_SIZE_K': 64, 
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128, 
            'BLOCK_SIZE_N': 32, 
            'BLOCK_SIZE_K': 64, 
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 32, 
            'BLOCK_SIZE_N': 128, 
            'BLOCK_SIZE_K': 64, 
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128, 
            'BLOCK_SIZE_N': 256, 
            'BLOCK_SIZE_K': 32, 
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 256, 
            'BLOCK_SIZE_N': 128, 
            'BLOCK_SIZE_K': 32, 
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 256, 
            'BLOCK_SIZE_N': 64, 
            'BLOCK_SIZE_K': 32, 
            'SPLIT_K': 1
        }, num_stages=5, num_warps=2),
        triton.Config({
            'BLOCK_SIZE_M': 64, 
            'BLOCK_SIZE_N': 256, 
            'BLOCK_SIZE_K': 32, 
            'SPLIT_K': 1
        }, num_stages=5, num_warps=2),
        triton.Config({
            'BLOCK_SIZE_M': 128, 
            'BLOCK_SIZE_N': 128, 
            'BLOCK_SIZE_K': 32, 
            'SPLIT_K': 1
        }, num_stages=5, num_warps=2),
        triton.Config({
            'BLOCK_SIZE_M': 128, 
            'BLOCK_SIZE_N': 64, 
            'BLOCK_SIZE_K': 32, 
            'SPLIT_K': 1
        }, num_stages=5, num_warps=2),
        triton.Config({
            'BLOCK_SIZE_M': 64, 
            'BLOCK_SIZE_N': 128, 
            'BLOCK_SIZE_K': 32, 
            'SPLIT_K': 1
        }, num_stages=5, num_warps=2),
        triton.Config({
            'BLOCK_SIZE_M': 128, 
            'BLOCK_SIZE_N': 32, 
            'BLOCK_SIZE_K': 32, 
            'SPLIT_K': 1
        }, num_stages=5, num_warps=2),
        triton.Config({
            'BLOCK_SIZE_M': 32, 
            'BLOCK_SIZE_N': 128, 
            'BLOCK_SIZE_K': 32, 
            'SPLIT_K': 1
        }, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr,
    ACC_TYPE: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    block_offset_m = pid_m * BLOCK_SIZE_M
    block_offset_n = pid_n * BLOCK_SIZE_N
    a_offs = block_offset_m[:, None] * stride_am + (tl.arange(0, BLOCK_SIZE_K)[None, :] * SPLIT_K)
    b_offs = (tl.arange(0, BLOCK_SIZE_K)[:, None] * SPLIT_K) * stride_bk +
