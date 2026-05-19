import torch
import triton
import triton.language as tl

from . import kernel_utils

@triton.jit
def matmul_kernel(
    c, a, b,
    M, N, K,
    stride_cm, stride_cn,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a + (
        offs_am[:, None] * stride_am +
        offs_k[None, :] * stride_ak
    )
    b_ptrs = b + (
        offs_k[:, None] * stride_bk +
        offs_bn[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        cur_a = tl.load(a_ptrs)
        cur_b = tl.load(b_ptrs)
        accumulator += tl.dot(cur_a, cur_b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_block = accumulator.to(tl.float16)
    tl.store(c_ptrs, c_block)

    
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix multiplication of `a` and `b` using Triton.
    The inputs `a` and `b` are assumed to be correctly reshaped and aligned
    with transpositions as necessary.
    """
    M, K = a.shape
    K, N = b.shape
    assert a.shape == (M, K)
    assert b.shape == (K, N)

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N'])
    )
    matmul_kernel[grid](
        c, a, b,
        M, N, K,
        c.stride(0), c.stride(1),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1)
    )
    return c
