import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8),
    ],
    key=['M', 'K']
)
@triton.jit
def quantize_int8_perrow_kernel(
    fpa_ptr,  # Pointer to the input floating-point matrix (MxK)
    a_ptr,    # Pointer to the output int8 matrix (MxK)
    as_ptr,   # Pointer to the output scale factors (M)
    M,        # Number of rows in the matrix
    K,        # Number of columns in the matrix
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start_m = pid * BLOCK_SIZE_M

    # Compute the maximum absolute value per row
    max_abs = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        block_start_k = k
        offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * K + tl.arange(0, BLOCK_SIZE_K)[None, :]
        mask = (block_start_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & (block_start_k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K)
        fpa = tl.load(fpa_ptr + block_start_m * K + block_start_k, mask=mask, other=0.0)
        max_abs = tl.max(max_abs, tl.abs(fpa), axis=1)

    # Compute the scale factor per row
    scale = 127.0 / max_abs

    # Quantize the matrix
    for k in range(0, K, BLOCK_SIZE_K):
        block_start_k = k
        offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * K + tl.arange(0, BLOCK_SIZE_K)[None, :]
        mask = (block_start_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & (block_start_k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K)
        fpa = tl.load(fpa_ptr + block_start_m * K + block_start_k, mask=mask, other=0.0)
        a = tl.round(fpa * scale[:, None])
        tl.store(a_ptr + block_start_m * K + block_start_k, a, mask=mask)

    # Store the scale factors
    tl.store(as_ptr + block_start_m, scale, mask=block_start_m + tl.arange(0, BLOCK_SIZE_M) < M)

import torch

def quantize_int8_perrow(fpa):
    M, K = fpa.shape
    a = torch.empty((M, K), dtype=torch.int8, device=fpa.device)
    as_ = torch.empty((M,), dtype=torch.float32, device=fpa.device)

    grid = (triton.cdiv(M, 128),)
    quantize_int8_perrow_kernel[grid](fpa, a, as_, M, K, BLOCK_SIZE_M=128, BLOCK_SIZE_K=128)

    return a, as_

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    a_ptr,    # Pointer to the int8 matrix A (MxK)
    as_ptr,   # Pointer to the scale factors for A (M)
    b_ptr,    # Pointer to the int8 matrix B (KxN)
    bs_ptr,   # Pointer to the scale factors for B (N)
    c_ptr,    # Pointer to the output matrix C (MxN)
    M,        # Number of rows in matrix A
    N,        # Number of columns in matrix B
    K,        # Number of columns in matrix A and rows in matrix B
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start_m = pid * BLOCK_SIZE_M

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K * SPLIT_K):
        block_start_k = k
        a = tl.load(a_ptr + block_start_m * K + block_start_k, mask=block_start_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M, other=0)
        b = tl.load(b_ptr + block_start_k * N, mask=block_start_k + tl.arange(0, BLOCK_SIZE_K)[:, None] < K, other=0)
        scale_a = tl.load(as_ptr + block_start_m, mask=block_start_m + tl.arange(0, BLOCK_SIZE_M) < M, other=1.0)
        scale_b = tl.load(bs_ptr, mask=block_start_k + tl.arange(0, BLOCK_SIZE_K) < K, other=1.0)

        for s in range(0, SPLIT_K):
            a_block = a[:, s * BLOCK_SIZE_K:(s + 1) * BLOCK_SIZE_K]
            b_block = b[s * BLOCK_SIZE_K:(s + 1) * BLOCK_SIZE_K, :]
            accumulator += tl.dot(a_block, b_block) * scale_a[:, None] * scale_b[None, :]

    tl.store(c_ptr + block_start_m * N, accumulator, mask=block_start_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M)

def matmul_quantize_int8(fpa, fpb):
    a, as_ = quantize_int8_perrow(fpa)
    b, bs_ = quantize_int8_perrow(fpb)
    return matmul_int8(a, as_, b, bs_)

def matmul_int8(a, as_, b, bs_):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)

    grid = (triton.cdiv(M, 128),)
    matmul_kernel[grid](a, as_, b, bs_, c, M, N, K, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, SPLIT_K=1)

    return c

def quantize_int8(fpa, axis=0):
    if axis == 0:
        return quantize_int8_perrow(fpa)
    else:
        raise NotImplementedError("Only row-wise quantization is supported for now")

# Example usage
M, K, N = 1024, 1024, 1024
fpa = torch.randn((M, K), dtype=torch.float32, device='cuda')
fpb = torch.randn((K, N), dtype=torch.float32, device='cuda')

# Quantize and perform matrix multiplication
c = matmul_quantize_int8(fpa, fpb)
print(c)
