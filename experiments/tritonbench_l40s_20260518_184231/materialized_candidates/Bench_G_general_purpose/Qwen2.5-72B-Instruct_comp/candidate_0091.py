import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr, scales_ptr, zero_points_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales, stride_zero_points,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
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

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    A = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    B = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    C = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_ptrs = A_ptr + (offs_am[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b_ptrs = B_ptr + ((k + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn)

        scales_ptrs = scales_ptr + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) * stride_scales
        zero_points_ptrs = zero_points_ptr + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) * stride_zero_points

        a_mask = (offs_am[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (k + offs_k[:, None] < K) & (offs_bn[None, :] < N)

        A = tl.load(a_ptrs, mask=a_mask, other=0.0)
        B = tl.load(b_ptrs, mask=b_mask, other=0.0)

        scales = tl.load(scales_ptrs)
        zero_points = tl.load(zero_points_ptrs)

        B = (B.to(tl.int32) - zero_points) * scales

        C += tl.dot(A, B)

    c_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)

    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, C, mask=c_mask)
    else:
        tl.store(c_ptrs, C, mask=c_mask)

import torch
import numpy as np

def quantize_int4(weight: torch.Tensor):
    weight = weight.to(torch.float32)
    max_val = torch.max(torch.abs(weight))
    scales = max_val / 7.0
    zero_points = torch.zeros_like(scales)
    quantized = torch.round(weight / scales).to(torch.int8)
    quantized = quantized.view(torch.int32)
    quantized = quantized << (4 * (torch.arange(quantized.numel()) % 2))
    quantized = quantized.view(weight.shape[0], -1)
    return quantized, scales, zero_points

def unpack_int4(quantized: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor):
    quantized = quantized.view(torch.int32)
    quantized = (quantized >> (4 * (torch.arange(quantized.numel()) % 2))) & 0xF
    quantized = quantized.view(quantized.shape[0], -1)
    dequantized = (quantized.to(torch.float32) - zero_points) * scales
    return dequantized

def matmul_dequantize_int4_s2(A: torch.Tensor, B_quantized: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor):
    M, K = A.shape
    N = B_quantized.shape[1]
    C = torch.zeros((M, N), device=A.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul_kernel[grid](
        A, B_quantized, C, scales, zero_points,
        M, N, K,
        A.stride(0), A.stride(1),
        B_quantized.stride(0), B_quantized.stride(1),
        C.stride(0), C.stride(1),
        scales.stride(0), zero_points.stride(0),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8, SPLIT_K=1
    )

    return C

# Example usage
A = torch.randn(1024, 512, device='cuda')
B = torch.randn(512, 1024, device='cuda')

B_quantized, scales, zero_points = quantize_int4(B)
C = matmul_dequantize_int4_s2(A, B_quantized, scales, zero_points)

# Verify the result
C_ref = torch.matmul(A, B)
print("Max error:", torch.max(torch.abs(C - C_ref)))
