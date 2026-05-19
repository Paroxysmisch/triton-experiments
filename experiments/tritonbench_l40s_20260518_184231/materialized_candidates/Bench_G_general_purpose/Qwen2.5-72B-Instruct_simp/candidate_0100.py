import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) * stride_am
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) * stride_bn
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] + offs_bn[None, :]) * stride_bk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_block = tl.load(a_ptrs)
        b_block = tl.load(b_ptrs)
        accumulator += tl.dot(a_block, b_block)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + (offs_am[:, None] + offs_bn[None, :]) * stride_cn
    if ACTIVATION == 1:
        accumulator = tl.where(accumulator > 0, accumulator, 0.01 * accumulator)
    tl.store(c_ptrs, accumulator)

import triton
import triton.runtime as triton_runtime
import numpy as np

def matmul(a, b, activation=False):
    M, K = a.shape
    K, N = b.shape
    c = np.empty((M, N), dtype=a.dtype)

    a_ptr = triton_runtime.to_device(a)
    b_ptr = triton_runtime.to_device(b)
    c_ptr = triton_runtime.to_device(c)

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    matmul_kernel[grid](
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        a.strides[0] // a.itemsize, a.strides[1] // a.itemsize,
        b.strides[0] // b.itemsize, b.strides[1] // b.itemsize,
        c.strides[0] // c.itemsize, c.strides[1] // c.itemsize,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        1 if activation else 0
    )

    return c
