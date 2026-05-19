import triton
import triton.language as tl

@triton.jit
def tril_mm_and_scale_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, P,
    alpha, beta,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(P, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_n)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None]
    offs_an = tl.arange(0, BLOCK_SIZE_K)[None, :]
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:, None]
    offs_bm = tl.arange(0, BLOCK_SIZE_K)[None, :]
    A = tl.load(A_ptr + (offs_am * stride_am + offs_an * stride_an), mask=offs_am < M, other=0.0)
    B = tl.load(B_ptr + (offs_bm * stride_bm + offs_bn * stride_bn), mask=offs_bn < P, other=0.0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, N, BLOCK_SIZE_K):
        A = tl.load(A_ptr + (offs_am * stride_am + (k + offs_an) * stride_an), mask=offs_am < M, other=0.0)
        B = tl.load(B_ptr + ((k + offs_bm) * stride_bm + offs_bn * stride_bn), mask=offs_bn < P, other=0.0)
        acc += tl.dot(A, B)
    
    acc *= alpha
    acc *= beta

    offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None]
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:, None]
    C = tl.load(C_ptr + (offs_cm * stride_cm + offs_cn * stride_cn), mask=offs_cm < M, other=0.0)
    C += acc
    tl.store(C_ptr + (offs_cm * stride_cm + offs_cn * stride_cn), C, mask=offs_cm < M)

import torch
import triton
import triton.language as tl

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert A.shape[1] == B.shape[0], "A and B must have compatible shapes for multiplication"

    M, N = A.shape
    P = B.shape[1]
    C = torch.empty((M, P), device=A.device, dtype=A.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(P, META['BLOCK_SIZE_N']),
    )

    tril_mm_and_scale_kernel[grid](
        A, B, C,
        M, N, P,
        alpha, beta,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16, GROUP_SIZE_M=8
    )

    return C

import torch

# Test data
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device='cuda')
B = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32, device='cuda')
alpha = 2.0
beta = 0.5

# Expected result using PyTorch
expected = beta * (alpha * torch.mm(torch.tril(A), B))

# Actual result using Triton
actual = tril_mm_and_scale(A, B, alpha, beta)

# Check if the results are close
print("Expected:\n", expected)
print("Actual:\n", actual)
print("Are the results close?", torch.allclose(expected, actual))
