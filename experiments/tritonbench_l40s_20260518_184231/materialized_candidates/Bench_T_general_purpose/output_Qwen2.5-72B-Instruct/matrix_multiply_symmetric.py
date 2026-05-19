import triton
import triton.language as tl

@triton.jit
def matrix_multiply_symmetric_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_a0, stride_a1,
    stride_b0, stride_b1,
    stride_c0, stride_c1,
    M, N, K,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    A = tl.load(A_ptr + (offs_am[:, None] * stride_a0 + offs_k[None, :] * stride_a1))
    B = tl.load(B_ptr + (offs_k[:, None] * stride_b0 + offs_bn[None, :] * stride_b1))
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        A = tl.load(A_ptr + (offs_am[:, None] * stride_a0 + (k + offs_k)[None, :] * stride_a1))
        B = tl.load(B_ptr + ((k + offs_k)[:, None] * stride_b0 + offs_bn[None, :] * stride_b1))
        acc += tl.dot(A, B)
    acc = alpha * acc
    C = tl.load(C_ptr + (offs_am[:, None] * stride_c0 + offs_bn[None, :] * stride_c1))
    C = acc + beta * C
    tl.store(C_ptr + (offs_am[:, None] * stride_c0 + offs_bn[None, :] * stride_c1), C)

    # Second operation: C = alpha * torch.mm(C, C.T) + beta * C
    C_T = tl.load(C_ptr + (offs_bn[:, None] * stride_c0 + offs_am[None, :] * stride_c1))
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, N, BLOCK_SIZE_N):
        C = tl.load(C_ptr + (offs_am[:, None] * stride_c0 + (k + offs_k)[None, :] * stride_c1))
        C_T = tl.load(C_ptr + ((k + offs_k)[:, None] * stride_c0 + offs_am[None, :] * stride_c1))
        acc += tl.dot(C, C_T)
    acc = alpha * acc
    C = tl.load(C_ptr + (offs_am[:, None] * stride_c0 + offs_bn[None, :] * stride_c1))
    C = acc + beta * C
    tl.store(C_ptr + (offs_am[:, None] * stride_c0 + offs_bn[None, :] * stride_c1), C)

import torch
import triton
import triton.language as tl

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on the same CUDA device"
    assert A.dtype == B.dtype == C.dtype, "All tensors must have the same data type"
    assert A.shape[1] == B.shape[0], "Matrix dimensions must be compatible for multiplication"
    assert A.shape[0] == C.shape[0] and B.shape[1] == C.shape[1], "Matrix C must have the correct dimensions"

    M, K = A.shape
    K, N = B.shape

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matrix_multiply_symmetric_kernel[grid](
        A, B, C,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        M, N, K,
        alpha, beta,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16, GROUP_SIZE_M=8
    )

    return C

# Example usage
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
B = torch.tensor([[0.5, -1.0], [1.5, 2.0]], device='cuda')
C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
alpha, beta = 2.0, 0.5
result = matrix_multiply_symmetric(A, B, C, alpha, beta)
print(result)
