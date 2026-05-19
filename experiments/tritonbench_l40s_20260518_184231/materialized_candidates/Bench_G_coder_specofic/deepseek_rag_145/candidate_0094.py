import triton.language as tl
import triton
import numpy as np

BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
BLOCK_SIZE_K = 16

@triton.jit
def matmul_kernel(c, a, b, M, N, K, stride_cm, stride_cn, stride_am, stride_ak, stride_bk, stride_bn):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    block_m, block_n = tl.program_id(0) * BLOCK_SIZE_M, tl.program_id(1) * BLOCK_SIZE_N

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_block = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_block = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a_block, b_block)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c[block_m : block_m + BLOCK_SIZE_M, block_n : block_n + BLOCK_SIZE_N] = accumulator


def matmul(c, a, b, M, N, K):
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    stride_cm, stride_cn = c.strides
    stride_am, stride_ak = a.strides
    stride_bk, stride_bn = b.strides
    matmul_kernel[grid_m, grid_n](c, a, b, M, N, K, stride_cm, stride_cn, stride_am, stride_ak, stride_bk, stride_bn)
