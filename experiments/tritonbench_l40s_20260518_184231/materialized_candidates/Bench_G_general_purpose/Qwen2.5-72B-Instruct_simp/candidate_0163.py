import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C, M, N, K, 
    stride_am, stride_ak, 
    stride_bk, stride_bn, 
    stride_cm, stride_cn, 
    alpha, beta, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, 
    GROUP_SIZE_M: tl.constexpr, 
    APPLY_ACTIVATION: tl.constexpr
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

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) * stride_am
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) * stride_bn
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_am[:, None] + k * stride_ak) < (M * K)
        b_mask = (offs_k[:, None] + k * stride_bn) < (K * N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bn

    c = accumulator.to(C.dtype.element_ty)
    if APPLY_ACTIVATION:
        c = tl.where(c > 0, c, c * alpha)

    offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) * stride_cm
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) * stride_cn
    c_ptrs = C + offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] + offs_cn[None, :]) < (M * N)
    tl.store(c_ptrs, c, mask=c_mask)

import torch
import triton
import triton.language as tl

# Define the grid and block sizes
BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
BLOCK_SIZE_K = 16
GROUP_SIZE_M = 8

# Define the kernel launch configuration
def matmul(A, B, C, alpha=0.01, beta=0.01, apply_activation=False):
    M, K = A.shape
    K, N = B.shape
    assert K == B.shape[0], "Matrix dimensions must match for multiplication"
    assert A.is_contiguous(), "Matrix A must be contiguous"
    assert B.is_contiguous(), "Matrix B must be contiguous"
    assert C.is_contiguous(), "Matrix C must be contiguous"

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul_kernel[grid](
        A, B, C, M, N, K, 
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1), 
        alpha, beta, 
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K, 
        GROUP_SIZE_M=GROUP_SIZE_M, 
        APPLY_ACTIVATION=apply_activation
    )

# Example usage
M, N, K = 1024, 1024, 1024
A = torch.randn((M, K), device='cuda', dtype=torch.float32)
B = torch.randn((K, N), device='cuda', dtype=torch.float32)
C = torch.zeros((M, N), device='cuda', dtype=torch.float32)

matmul(A, B, C, apply_activation=True)

print(C)
