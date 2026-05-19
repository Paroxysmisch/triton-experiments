import triton
import triton.language as tl

@triton.jit
def matmul4_kernel(
    A, B, C, scales, qzeros,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_scales_n, stride_qzeros_n,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = num_pid_m
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Dequantize B
        b_int32 = tl.cast(b, tl.int32)
        b_4bit = (b_int32 >> (4 * (offs_k[None, :] % 8))) & 0xF
        b_dequant = tl.cast(b_4bit, tl.float32)
        b_dequant = b_dequant * tl.load(scales + (offs_bn[None, :] // 8) * stride_scales_n + (offs_k[None, :] // 8))
        b_dequant = b_dequant - tl.load(qzeros + (offs_bn[None, :] // 8) * stride_qzeros_n + (offs_k[None, :] // 8))

        acc += tl.dot(a, b_dequant)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

import torch

def matmul_dequantize_int4_gptq(x, qweight, scales, qzeros, out=None):
    M, K = x.shape
    _, N = qweight.shape
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32

    if out is None:
        out = torch.empty((M, N), dtype=torch.float16, device=x.device)

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    matmul4_kernel[grid](
        x, qweight, out, scales, qzeros,
        x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1), out.stride(0), out.stride(1),
        scales.stride(0), qzeros.stride(0),
        M, N, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    return out

@triton.jit
def matmul_kernel(
    A, B, C, scales, qzeros,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_scales_n, stride_qzeros_n,
    M, N, K, SPLIT_K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = num_pid_m
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K * SPLIT_K):
        for s in range(SPLIT_K):
            a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
            b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)

            # Dequantize B
            b_int32 = tl.cast(b, tl.int32)
            b_4bit = (b_int32 >> (4 * (offs_k[None, :] % 8))) & 0xF
            b_dequant = tl.cast(b_4bit, tl.float32)
            b_dequant = b_dequant * tl.load(scales + (offs_bn[None, :] // 8) * stride_scales_n + (offs_k[None, :] // 8))
            b_dequant = b_dequant - tl.load(qzeros + (offs_bn[None, :] // 8) * stride_qzeros_n + (offs_k[None, :] // 8))

            acc += tl.dot(a, b_dequant)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.atomic_add(c_ptrs, c, mask=c_mask)
