import torch
import triton
import triton.language as tl
import numpy as np
from numpy.random import RandomState
import time

@triton.jit
def iv_dependent_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        type: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = num_pid_n * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, (pid + 1) // num_pid_in_group - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if type == "preload_1":
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c = tl.dot(a, b)
    elif type == "preload_2":
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c = tl.dot(a, b)
    elif type == "no_preload":
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c = tl.dot(a, b)
    elif type == "early_preload":
        a = tl.load(a_ptrs)
        c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K) - 1):
            b = tl.load(b_ptrs)
            c += tl.dot(a, b)
            a = tl.load(a_ptrs + (BLOCK_SIZE_K * (k + 1) * stride_ak))
    elif type == "late_preload":
        a = tl.load(a_ptrs)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K) - 1):
            b = tl.load(b_ptrs)
            c = tl.dot(a, b)
            a = tl.load(a_ptrs + (BLOCK_SIZE_K * (k + 1) * stride_ak))
    elif type == "intra_tiles":
        for k in range(0, BLOCK_SIZE_K):
            a = tl.load(a_ptrs + k * stride_ak)
            b = tl.load(b_ptrs + k * stride_bk)
            c += tl.dot(a, b)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def iv_dependent_matmul_wrapper(
        M, N, K,
        NUM_CTAS, NUM_WARPS,
        GROUP_SIZE_M,
        type
):
    if type == "intra_tiles":
        assert (K % BLOCK_SIZE_K == 0)

    a = torch.rand((M, K), device='cuda', dtype=torch.float16)
    b = torch.rand((K, N), device='cuda', dtype=torch.float16)
    c = torch.zeros((M, N), device='cuda', dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    iv_dependent_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
        type=type
    )
    return c
