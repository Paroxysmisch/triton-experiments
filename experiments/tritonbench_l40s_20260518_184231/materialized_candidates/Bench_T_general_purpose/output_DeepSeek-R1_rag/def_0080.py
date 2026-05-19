import torch
import triton
import triton.language as tl
import math

# Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# Triton kernel for upper triangular solve
@triton.jit
def tri_solve_kernel(R_ptr, B_ptr, X_ptr, n, k,
                     R_row_stride, R_col_stride,
                     B_row_stride, B_col_stride,
                     X_row_stride, X_col_stride,
                     BLOCK_SIZE: tl.constexpr):
    col_idx = tl.program_id(0)
    if col_idx >= k:
        return
    for i in range(n-1, -1, -1):
        R_ii = tl.load(R_ptr + i * R_row_stride + i * R_col_stride)
        sum_val = 0.0
        for j in range(i + 1, n):
            R_ij = tl.load(R_ptr + i * R_row_stride + j * R_col_stride)
            X_j = tl.load(X_ptr + j * X_row_stride + col_idx * X_col_stride)
            sum_val += R_ij * X_j
        B_i = tl.load(B_ptr + i * B_row_stride + col_idx * B_col_stride)
        X_i = (B_i - sum_val) / R_ii
        tl.store(X_ptr + i * X_row_stride + col_idx * X_col_stride, X_i)

def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # QR decomposition using PyTorch
    m, n = A.shape
    Q, R = torch.linalg.qr(A)
    k = b.shape[1]
    device = A.device

    # Compute Q^T * b using Triton matmul
    Q_T = Q.T
    Q_T_b = torch.empty((n, k), device=device, dtype=A.dtype)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    grid_m = triton.cdiv(n, BLOCK_SIZE_M)
    grid_n = triton.cdiv(k, BLOCK_SIZE_N)
    matmul_kernel[(grid_m, grid_n, 1)](
        Q_T, b, Q_T_b,
        M=n, N=k, K=m,
        stride_am=Q_T.stride(0), stride_ak=Q_T.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=Q_T_b.stride(0), stride_cn=Q_T_b.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    # Solve R x = Q_T_b using Triton triangular solve
    x = torch.zeros_like(Q_T_b)
    tri_solve_kernel[(triton.cdiv(k, 1),)](
        R, Q_T_b, x,
        n=n, k=k,
        R_row_stride=R.stride(0), R_col_stride=R.stride(1),
        B_row_stride=Q_T_b.stride(0), B_col_stride=Q_T_b.stride(1),
        X_row_stride=x.stride(0), X_col_stride=x.stride(1),
        BLOCK_SIZE=1
    )
    return x
