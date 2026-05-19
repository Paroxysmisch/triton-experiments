import triton
import triton.language as tl

# Constants for block sizes
BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
BLOCK_SIZE_K = 16
GROUP_SIZE_M = 8

# Matmul4 Kernel
@triton.jit
def matmul4_kernel(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize, NO_GROUPS,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
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
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :])

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a_block = tl.load(a_ptrs)
        b_block = tl.load(b_ptrs)
        b_block = b_block.to(tl.float32)
        accumulator += tl.dot(a_block, b_block)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# Dequantize Kernel
@triton.jit
def dequantize_kernel(
    b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
    K, N, group_size,
    stride_bk, stride_bn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    stride_fpbk, stride_fpbn,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_k = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    b_ptrs = b_ptr + stride_bk * offs_k[:, None] + stride_bn * offs_n[None, :]
    b_scale_ptrs = b_scale_ptr + stride_bsk * offs_k[:, None] + stride_bsn * offs_n[None, :]
    b_zp_ptrs = b_zp_ptr + stride_bzpk * offs_k[:, None] + stride_bzpn * offs_n[None, :]

    b_block = tl.load(b_ptrs)
    b_scale_block = tl.load(b_scale_ptrs)
    b_zp_block = tl.load(b_zp_ptrs)

    fpb_block = (b_block.to(tl.float32) - b_zp_block.to(tl.float32)) * b_scale_block
    fpb_ptrs = fpb_ptr + stride_fpbk * offs_k[:, None] + stride_fpbn * offs_n[None, :]
    tl.store(fpb_ptrs, fpb_block)

# Helper Function: Dequantize Int4
def dequantize_int4(b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr, K, N, group_size, stride_bk, stride_bn, stride_bsk, stride_bsn, stride_bzpk, stride_bzpn, stride_fpbk, stride_fpbn):
    grid = lambda META: (tl.cdiv(K, META['BLOCK_SIZE_K']) * tl.cdiv(N, META['BLOCK_SIZE_N']),)
    dequantize_kernel[grid](b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr, K, N, group_size, stride_bk, stride_bn, stride_bsk, stride_bsn, stride_bzpk, stride_bzpn, stride_fpbk, stride_fpbn, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N)

# Wrapper Function: Matmul Dequantize Int4 S1
def matmul_dequantize_int4_s1(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales_g, stride_scales_n, stride_zeros_g, stride_zeros_n, groupsize, NO_GROUPS):
    fpb_ptr = triton.empty((K, N), dtype=triton.float32)
    dequantize_int4(b_ptr, scales_ptr, zeros_ptr, fpb_ptr, K, N, groupsize, stride_bk, stride_bn, stride_scales_g, stride_scales_n, stride_zeros_g, stride_zeros_n, stride_fpbk=1, stride_fpbn=K)
    grid = lambda META: (tl.cdiv(M, META['BLOCK_SIZE_M']) * tl.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul4_kernel[grid](a_ptr, fpb_ptr, c_ptr, scales_ptr, zeros_ptr, M, N, K, stride_am, stride_ak, 1, 1, stride_cm, stride_cn, stride_scales_g, stride_scales_n, stride_zeros_g, stride_zeros_n, groupsize, NO_GROUPS, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M)

# Helper Function: Quantize Int4
def quantize_int4(weights, group_size):
    K, N = weights.shape
    num_groups = (K + group_size - 1) // group_size
    scales = triton.empty((num_groups, N), dtype=triton.float32)
    zeros = triton.empty((num_groups, N), dtype=triton.float32)
    quantized_weights = triton.empty((K, N), dtype=triton.int4)

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, K)
        group_weights = weights[start:end, :]
        min_val = triton.min(group_weights, axis=0)
        max_val = triton.max(group_weights, axis=0)
        scales[g, :] = (max_val - min_val) / 15.0
        zeros[g, :] = triton.round(min_val / scales[g, :])
        quantized_group = triton.round((group_weights - min_val) / scales[g, :])
        quantized_weights[start:end, :] = quantized_group.to(triton.int4)

    return quantized_weights, scales, zeros
