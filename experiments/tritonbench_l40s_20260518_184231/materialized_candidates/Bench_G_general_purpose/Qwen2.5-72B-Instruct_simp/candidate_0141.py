import triton
import triton.language as tl

@triton.jit
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, ACC_TYPE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am * stride_am + offs_k * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k in range(0, K, BLOCK_K):
        a_mask = (offs_am < M) & (offs_k < K)
        b_mask = (offs_k[:, None] < K) & (offs_bn < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float32)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    tl.store(c_ptrs, c, mask=c_mask)

import torch
import triton
import triton.language as tl

def matmul_persistent(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_SIZE_M=8):
    # Ensure the input tensors are on the same device
    device = a.device
    assert a.device == b.device, "Input tensors must be on the same device"

    # Ensure the input tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()

    # Ensure the input dimensions are compatible
    M, K = a.shape
    K, N = b.shape
    assert K == b.shape[0], "Incompatible dimensions for matrix multiplication"

    # Allocate memory for the output matrix
    c = torch.empty((M, N), device=device, dtype=a.dtype)

    # Define the grid configuration
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    # Determine the accumulator type based on the input data type
    if a.dtype == torch.float16:
        ACC_TYPE = tl.float32
    elif a.dtype == torch.float8:
        ACC_TYPE = tl.float16
    else:
        raise ValueError("Unsupported data type")

    # Launch the kernel
    matmul_kernel_persistent[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        GROUP_SIZE_M, ACC_TYPE
    )

    return c
