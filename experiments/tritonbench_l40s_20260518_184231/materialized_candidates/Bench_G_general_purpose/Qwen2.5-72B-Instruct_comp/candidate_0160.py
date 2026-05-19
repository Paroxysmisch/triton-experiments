import triton
import triton.language as tl

@triton.jit
def matmul4_kernel(
    A, B, C, scales, zeros, M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
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
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k)
        b_mask = (offs_k[:, None] < K - k) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Dequantize B
        b_int4 = b.to(tl.int32)
        b_int4 = (b_int4 >> (tl.arange(0, 8) * 4)) & 0xF
        b_float = (b_int4 - zeros[offs_bn[None, :]]) * scales[offs_bn[None, :]]

        accumulator += tl.dot(a, b_float)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

import torch

def matmul_dequantize_int4_gptq(A, B, scales, zeros, C=None, in_place=False):
    M, K = A.shape
    _, N = B.shape
    if C is None:
        C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    elif not in_place:
        C = C.clone()

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul4_kernel[grid](
        A, B, C, scales, zeros, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8
    )

    return C

def quantize_int4(B, group_size=128):
    B = B.to(torch.float32)
    B = B.t().contiguous()
    B = B.view(-1, group_size)
    min_vals, _ = B.min(dim=1)
    max_vals, _ = B.max(dim=1)
    scales = (max_vals - min_vals) / 15
    zeros = min_vals / scales
    B_quantized = (B / scales[:, None] + zeros[:, None]).round().clamp(-8, 7).to(torch.int32)
    B_quantized = B_quantized.view(-1, 8)
    B_quantized = (B_quantized[:, 0] << 28) | (B_quantized[:, 1] << 24) | (B_quantized[:, 2] << 20) | (B_quantized[:, 3] << 16) | (B_quantized[:, 4] << 12) | (B_quantized[:, 5] << 8) | (B_quantized[:, 6] << 4) | B_quantized[:, 7]
    return B_quantized, scales, zeros
