import triton
import triton.language as tl
import torch

# Triton kernel for dequantizing packed int4 matrix
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_N': 128}, num_warps=8),
    ],
    key=['K', 'N']
)
@triton.jit
def dequantize_kernel(
    b_ptr, fpb_ptr, scales_ptr, zeros_ptr,
    K, N,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    first_pid_n = group_id * num_pid_n
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = first_pid_n + (pid % num_pid_n)

    offs_k = pid_m * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    b_offs = (offs_k // 8) * N + offs_n
    fpb_offs = offs_k * N + offs_n

    b = tl.load(b_ptr + b_offs)
    scales = tl.load(scales_ptr + b_offs)
    zeros = tl.load(zeros_ptr + b_offs)

    b_unpacked = tl.where(offs_k % 8 < 4, (b >> (4 * (offs_k % 8 // 4))) & 0xF, (b >> (4 * ((offs_k % 8 // 4) - 1))) & 0xF)
    fpb = (b_unpacked - zeros) * scales

    tl.store(fpb_ptr + fpb_offs, fpb)

# Triton kernel for matrix multiplication with packed int4 matrix
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul4_kernel(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    first_pid_n = group_id * num_pid_n
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = first_pid_n + (pid % num_pid_n)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k // 8)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        b_unpacked = tl.where(offs_k % 8 < 4, (b >> (4 * (offs_k % 8 // 4))) & 0xF, (b >> (4 * ((offs_k % 8 // 4) - 1))) & 0xF)
        scales = tl.load(scales_ptr + b_ptrs, mask=b_mask, other=1.0)
        zeros = tl.load(zeros_ptr + b_ptrs, mask=b_mask, other=0.0)

        b_unpacked = (b_unpacked - zeros) * scales

        accumulator += tl.dot(a, b_unpacked)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk

    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# Wrapper function for dequantizing packed int4 matrix
def dequantize_int4(b, scales, zeros):
    M, N = b.shape
    K = M * 8
    fpb = torch.empty((K, N), dtype=torch.float16, device=b.device)
    grid = lambda META: (triton.cdiv(K, META['BLOCK_SIZE_K']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    dequantize_kernel[grid](b, fpb, scales, zeros, K, N)
    return fpb

# Wrapper function for matrix multiplication with dequantized int4 matrix
def matmul_dequantize_int4_s1(a, b, scales, zeros):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul4_kernel[grid](a, b, c, scales, zeros, M, N, K)
    return c

# Function for quantizing weights into int4 format
def quantize_int4(weights, group_size=128):
    assert weights.dtype == torch.float16, "Input weights must be float16"
    K, N = weights.shape
    assert K % group_size == 0, "K must be divisible by group_size"

    scales = torch.empty((K // group_size, N), dtype=torch.float16, device=weights.device)
    zeros = torch.empty((K // group_size, N), dtype=torch.float16, device=weights.device)
    b = torch.empty((K // 8, N), dtype=torch.int32, device=weights.device)

    for i in range(0, K, group_size):
        group = weights[i:i + group_size, :]
        min_val, _ = torch.min(group, dim=0)
        max_val, _ = torch.max(group, dim=0)
        scale = (max_val - min_val) / 15
        zero = -min_val / scale
        scales[i // group_size, :] = scale
        zeros[i // group_size, :] = zero
        quantized = torch.round((group / scale) + zero)
        quantized = quantized.to(torch.int32)
        b[i // 8, :] = (quantized[0::2] << 4) | quantized[1::2]

    return b, scales, zeros
