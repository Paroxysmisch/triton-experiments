import torch
import triton
import triton.language as tl

@triton.jit
def matmul4_kernel(a, b, c, scales, qzeros, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                   GROUP_SIZE_M: tl.constexpr):
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
    a_ptrs = a + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        b = (b >> 4) & 0xf
        b = (b & 0x8) - (b & 0x7)
        zero_ptrs = qzeros + offs_k
        zero = tl.load(zero_ptrs, mask=offs_k < K - k * BLOCK_SIZE_K, other=0.0)
        b = b + zero
        b = b.to(tl.float32)
        scales_ptrs = scales + offs_k
        scale = tl.load(scales_ptrs, mask=offs_k < K - k * BLOCK_SIZE_K, other=0.0)
        b = b * scale
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@torch.no_grad()
def matmul_dequantize_int4_gptq(x, qweight, scales, qzeros):
    device = x.device
    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()
    if qweight.stride(0) > 1 and qweight.stride(1) > 1:
        qweight = qweight.contiguous()
    M, K = x.shape
    N, K = qweight.shape
    c = torch.empty((M, N), device=device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul4_kernel[grid](x, qweight, c, scales, qzeros, x.stride(0), x.stride(1), qweight.stride(0),
                         qweight.stride(1), c.stride(0), c.stride(1), M, N)
    return c


@triton.jit
def matmul_kernel(a, b, c, scales, qzeros, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr):
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
    offs_k = tl.arange(0, BLOCK_SIZE_K * SPLIT_K) // BLOCK_SIZE_K
    a_ptrs = a + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K * SPLIT_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * (BLOCK_SIZE_K * SPLIT_K), other=0.0)

        b = (b >> 4) & 0xf
        b = (b & 0x8) - (b & 0x7)
        zero_ptrs = qzeros + offs_k
        zero = tl.load(zero_ptrs, mask=offs_k < K - k * (BLOCK_SIZE_K * SPLIT_K), other=0.0)
        b = b + zero
        b = b.to(tl.float32)
        scales_ptrs = scales + offs_k
        scale = tl.load(scales_ptrs, mask=offs_k < K - k * (BLOCK_SIZE_K * SPLIT_K), other=0.0)
        b = b * scale
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.atomic_add(c_ptrs, c, mask=c_mask)


@torch.no_grad()
def matmul_dequantize_int4_s2(x, qweight, scales, qzeros):
    device = x.device
    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()
    if qweight.stride(0) > 1 and qweight.stride(1) > 1:
        qweight = qweight.contiguous()
    M, K = x.shape
    N, K = qweight.shape
    c = torch.empty((M, N), device=device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](x, qweight, c, scales, qzeros, x.stride(0), x.stride(1), qweight.stride(0),
                        qweight.stride(1), c.stride(0), c.stride(1), M, N, num_stages=1, SPLIT_K=2)
    return c


@triton.jit
def dequantize_kernel(b, c, stride_bk, stride_bn, stride_ck, stride_cn, M, N, K, BLOCK_SIZE_K: tl.constexpr,
                      BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_k = group_id * GROUP_SIZE_M
    group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE_M)
    pid_k = first_pid_k
