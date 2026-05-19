import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def iv_dependent_matmul_kernel(a, b, c,
    M, N, K,
    stride_a, stride_b, stride_c, 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    type: tl.constexpr):

    pid = tl.program_id(0)
    grid_size = tl.grid_size(0)

    # Calculate offsets for loading data
    block_start_m = pid * BLOCK_SIZE_M
    block_start_n = pid * BLOCK_SIZE_N

    # Load data with offsets
    a_block = tl.load(a + block_start_m * stride_a,
                      mask=(tl.arange(0, BLOCK_SIZE_M) + block_start_m < M))
    b_block = tl.load(b + block_start_n * stride_b,
                      mask=(tl.arange(0, BLOCK_SIZE_N) + block_start_n < N))

    # Compute matrix multiplication
    c_block = tl.dot(a_block, b_block)

    # Store results
    c_offsets = pid * BLOCK_SIZE_M * stride_c + tl.arange(0, BLOCK_SIZE_M) * stride_c
    tl.store(c + c_offsets, c_block,
             mask=(tl.arange(0, BLOCK_SIZE_M) + block_start_m < M) &
                  (tl.arange(0, BLOCK_SIZE_N) + block_start_n < N))

def iv_dependent_matmul_wrapper(a, b, type="default", BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64):
    device = a.device
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=device, dtype=a.dtype)

    # Set grid size
    grid_size = (M * N + BLOCK_SIZE_M * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N)

    # Invoke kernel
    iv_dependent_matmul_kernel[grid_size](a, b, c,
                                          M, N, K,
                                          a.stride(0), a.stride(1), c.stride(0), c.stride(1),
                                          BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                          type)

    return c
