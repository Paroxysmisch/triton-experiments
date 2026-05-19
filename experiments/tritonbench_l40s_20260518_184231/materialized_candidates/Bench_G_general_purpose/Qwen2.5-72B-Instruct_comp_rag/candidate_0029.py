import torch
import triton
import triton.language as tl

@triton.jit
def quant_matmul_248_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, g_ptr,
    M, N, K,
    bits, maxq,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales, stride_zeros,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    infearure_per_bits = 32 // bits
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

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_mask = (offs_am[:, None] < M)
    b_ptrs = b_ptr + (
        (offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn
    )
    g_ptrs = g_ptr + offs_k
    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] // infearure_per_bits)

    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        g_idx = tl.load(g_ptrs)
        scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)
        zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros)
        zeros = (zeros >> zeros_shifter[None, :]) & maxq
        zeros = (zeros + 1)

        a = tl.load(a_ptrs, mask=a_mask, other=0.)
        b = tl.load(b_ptrs)

        b = (b >> shifter[:, None]) & maxq
        b = (b - zeros) * scales

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
        g_ptrs += BLOCK_SIZE_K

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def transpose_quant_matmul_248_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, g_ptr,
    M, N, K,
    bits, maxq,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales, stride_zeros,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    infearure_per_bits = 32 // bits
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_n[None, :] * stride_ak)
    a_mask = (offs_am[:, None] < M)
    b_ptrs = b_ptr + (
        (offs_bk[:, None] // infearure_per_bits) * stride_bk + offs_n[None, :] * stride_bn
    )
    g_ptrs = g_ptr + offs_bk
    g_idx = tl.load(g_ptrs)
    scales_ptrs = scales_ptr + offs_n[None, :] + g_idx[:, None] * stride_scales
    zeros_ptrs = zeros_ptr + (offs_n[None, :] // infearure_per_bits) + g_idx[:, None] * stride_zeros

    shifter = (offs_bk % infearure_per_bits) * bits
    zeros_shifter = (offs_n % infearure_per_bits) * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k in range(0, num_pid_n):
        scales = tl.load(scales_ptrs)
        zeros = tl.load(zeros_ptrs)
        zeros = (zeros >> zeros_shifter[None, :]) & maxq
        zeros = (zeros + 1)

        a = tl.load(a_ptrs, mask=a_mask, other=0.)
        b = tl.load(b_ptrs)

        b = (b >> shifter[:, None]) & maxq
        b = (b - zeros) * scales
        b = tl.trans(b)

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_N
        b_ptrs += BLOCK_SIZE_N
        scales_ptrs += BLOCK_SIZE_N
        zeros_ptrs += (BLOCK_SIZE_N // infearure_per_bits)

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bk[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bk[None, :] < K)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output = torch.empty((input.shape[0], qweight.shape[1]), device=input.device, dtype=input.dtype)
        grid = lambda META: (
            triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(qweight.shape[1], META['BLOCK_SIZE_N']),
        )
        quant_matmul_248_kernel[grid](
            input, qweight, output,
            scales.to(input.dtype), qzeros, g_idx,
            input.shape[0], qweight.shape[1], input.shape[1],
            bits, maxq,
            input.stride(0), input.stride(1),
            qweight.stride(0), qweight.stride(1),
            output.stride(0), output.stride(1),
            scales.stride(0), qzeros.stride(0)
        )
        return output


def transpose_quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output_dim = (qweight.shape[0] * 32) // bits
        output = torch.empty((input.shape[0], output_dim), device=input.device, dtype=input.dtype)
        grid = lambda META: (
            triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(output_dim, META['BLOCK_SIZE_K']),)
        transpose_quant_matmul_248_kernel[grid](
            input, qweight, output,
            scales.to(input.dtype), qzeros, g_idx,
            input.shape[0], qweight.shape[1], output_dim,
            bits, maxq,
            input.stride(0), input.stride(1),
            qweight.stride(0), qweight.stride(1),
            output.stride(0), output.stride(1),
            scales.stride(0), qzeros.stride(0)
        )
        return output
