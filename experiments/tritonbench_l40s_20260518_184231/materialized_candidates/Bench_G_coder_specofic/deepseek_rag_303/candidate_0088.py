import torch
import triton
import triton.language as tl

@triton.jit
def matmul4_kernel(
    a_ptr, b_ptr, qweight_ptr, scales_ptr, qzeros_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_qweight_am, stride_qweight_ak,
    stride_qweight_bn,
    stride_scales_k,
    stride_zeros_k,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m - first_pid_m, num_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_bn[None, :] * stride_ak)
    a_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)

    b_ptrs = b_ptr + (offs_am[:, None] * stride_am + offs_bn[None, :] * stride_ak)
    b_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    qweight_ptrs = qweight_ptr + (offs_am[:, None] * stride_qweight_am + offs_k[None, :] * stride_qweight_ak)
    qweight_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)

    scales_ptrs = scales_ptr + offs_k * stride_scales_k
    scales_mask = offs_k < K

    qzeros_ptrs = qzeros_ptr + offs_k * stride_zeros_k
    qzeros_mask = offs_k < K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        qweight = tl.load(qweight_ptrs, mask=qweight_mask, other=0)
        scales = tl.load(scales_ptrs, mask=scales_mask, other=0.0).to(tl.float32)
        qzeros = tl.load(qzeros_ptrs, mask=qzeros_mask, other=0.0).to(tl.float32)

        # Dequantize
        b = (b.to(tl.int32) & 0xf) * qweight
        scales = scales[:, None]
        qzeros = qzeros[:, None]

        # Compute: accumulator += a * b
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        qweight_ptrs += BLOCK_SIZE_K * stride_qweight_ak

    accumulator = accumulator.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul_dequantize_int4(x, qweight, scales, qzeros, group_size, out=None):
    M, K = x.shape
    Kw, N = qweight.shape

    if out is None:
        c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    else:
        c = out

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul4_kernel[grid](
        x, qweight, scales, qzeros, c,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0),
        qzeros.stride(0),
        c.stride(0), c.stride(1)
    )
    return c

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, qweight_ptr, scales_ptr, qzeros_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_qweight_am, stride_qweight_ak,
    stride_qweight_bn,
    stride_scales_k,
    stride_zeros_k,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    matrix multiplication:  C = A x B
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    qweight is of shape (M, K//8, N) int32
    scales is of shape (K) float16
    qzeros is of shape (K//8, N) int32
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m - first_pid_m, num_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + t
