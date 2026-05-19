import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'K']
)
@triton.jit
def quantize_int8_perrow_kernel(fpa_ptr, a_ptr, as_ptr, M, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    block_start_m = pid * BLOCK_SIZE_M

    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    mask_m = offsets_m < M
    mask_k = offsets_k < K

    fpa = tl.load(fpa_ptr + offsets_m[:, None] * K + offsets_k[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0)

    max_abs = tl.max(tl.abs(fpa), axis=1)
    scale = 127.0 / max_abs
    tl.store(as_ptr + offsets_m, scale, mask=mask_m)

    quantized = tl.round(fpa * scale[:, None]).to(tl.int8)
    tl.store(a_ptr + offsets_m[:, None] * K + offsets_k[None, :], quantized, mask=mask_m[:, None] & mask_k[None, :])

import torch

def quantize_int8_perrow(fpa):
    M, K = fpa.shape
    a = torch.empty((M, K), dtype=torch.int8, device=fpa.device)
    as_ = torch.empty(M, dtype=torch.float32, device=fpa.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),)
    quantize_int8_perrow_kernel[grid](fpa, a, as_, M, K)

    return a, as_

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 2}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    block_start_m = pid // (N // BLOCK_SIZE_N) * BLOCK_SIZE_M
    block_start_n = pid % (N // BLOCK_SIZE_N) * BLOCK_SIZE_N

    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptr + offsets_m[:, None] * K + (offsets_k[None, :] + k), mask=(offsets_m[:, None] < M) & (offsets_k[None, :] + k < K), other=0).to(tl.float32)
        b = tl.load(b_ptr + (offsets_k[:, None] + k) * N + offsets_n[None, :], mask=(offsets_k[:, None] + k < K) & (offsets_n[None, :] < N), other=0).to(tl.float32)

        acc += tl.dot(a, b)

    scale_a = tl.load(as_ptr + offsets_m, mask=offsets_m < M)
    scale_b = tl.load(bs_ptr + offsets_n, mask=offsets_n < N)

    acc *= scale_a[:, None] * scale_b[None, :]
    tl.store(c_ptr + offsets_m[:, None] * N + offsets_n[None, :], acc, mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N))

def matmul_int8(a, b, as_, bs_, M, N, K):
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](a, b, c, as_, bs_, M, N, K)

    return c

def matmul_quantize_int8(fpa, fpb):
    a, as_ = quantize_int8_perrow(fpa)
    b, bs_ = quantize_int8_perrow(fpb.T)  # Transpose B for correct multiplication shape

    M, K = a.shape
    _, N = b.shape

    return matmul_int8(a, b, as_, bs_, M, N, K)
