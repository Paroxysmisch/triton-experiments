import triton
import triton.language as tl
import torch

# Kernel for matrix multiplication with quantization adjustments
@triton.jit
def matmul4_kernel(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn

    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] // 8)

    c_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        scales = tl.load(scales_ptrs)
        zeros = tl.load(zeros_ptrs)

        b = (b >> ((offs_k % 8) * 4)) & 0xF
        b = (b - zeros) * scales

        c_acc += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk

    c = c_acc.to(tl.float16)
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptrs, c)

# Kernel for dequantizing a packed 4-bit integer matrix
@triton.jit
def dequantize_kernel(
    b_ptr, fpb_ptr, scales_ptr, zeros_ptr,
    K, N,
    stride_bk, stride_bn, stride_fpbk, stride_fpbn,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn
    fpb_ptrs = fpb_ptr + offs_k[:, None] * stride_fpbk + offs_bn[None, :] * stride_fpbn

    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] // 8)

    b = tl.load(b_ptrs)
    scales = tl.load(scales_ptrs)
    zeros = tl.load(zeros_ptrs)

    b = (b >> ((offs_k % 8) * 4)) & 0xF
    fpb = (b - zeros) * scales

    tl.store(fpb_ptrs, fpb)

# Wrapper function for dequantization
def dequantize_int4(b, scales, zeros, K, N):
    fpb = torch.empty((K, N), dtype=torch.float32, device=b.device)
    grid = lambda META: (triton.cdiv(K, META['BLOCK_SIZE_K']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    dequantize_kernel[grid](
        b, fpb, scales, zeros,
        K, N,
        b.stride(0), b.stride(1), fpb.stride(0), fpb.stride(1)
    )
    return fpb

# Wrapper function for matrix multiplication with dequantization
def matmul_dequantize_int4_s1(a, b, scales, zeros, M, N, K):
    fpb = dequantize_int4(b, scales, zeros, K, N)
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    matmul4_kernel[grid](
        a, fpb, c, scales, zeros,
        M, N, K,
        a.stride(0), a.stride(1), fpb.stride(0), fpb.stride(1), c.stride(0), c.stride(1)
    )
    return c
