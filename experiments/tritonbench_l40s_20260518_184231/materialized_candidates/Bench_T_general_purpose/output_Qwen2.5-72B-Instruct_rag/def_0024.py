import triton
import triton.language as tl

@triton.jit
def least_squares_qr_kernel(
    A_ptr,
    b_ptr,
    Q_ptr,
    R_ptr,
    x_ptr,
    m,
    n,
    k,
    stride_am,
    stride_an,
    stride_bk,
    stride_bn,
    stride_qm,
    stride_qn,
    stride_rn,
    stride_rk,
    stride_xn,
    stride_xk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Solves the least squares problem for an overdetermined system of linear equations using QR decomposition.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(k, BLOCK_SIZE_K)

    # Compute the QR decomposition
    for i in range(n):
        for j in range(i, n):
            if j == i:
                R_value = 0.0
                for l in range(i, m):
                    R_value += tl.load(A_ptr + l * stride_am + i * stride_an) * tl.load(A_ptr + l * stride_am + j * stride_an)
                R_value = tl.sqrt(R_value)
                tl.store(R_ptr + i * stride_rn + j * stride_rk, R_value)
            else:
                R_value = 0.0
                for l in range(i, m):
                    R_value += tl.load(A_ptr + l * stride_am + i * stride_an) * tl.load(A_ptr + l * stride_am + j * stride_an)
                R_value /= tl.load(R_ptr + i * stride_rn + i * stride_rk)
                tl.store(R_ptr + i * stride_rn + j * stride_rk, R_value)

        for j in range(i + 1, m):
            Q_value = 0.0
            for l in range(i, n):
                Q_value += tl.load(A_ptr + j * stride_am + l * stride_an) * tl.load(R_ptr + i * stride_rn + l * stride_rk)
            Q_value /= tl.load(R_ptr + i * stride_rn + i * stride_rk)
            for l in range(i, n):
                A_value = tl.load(A_ptr + j * stride_am + l * stride_an) - Q_value * tl.load(R_ptr + i * stride_rn + l * stride_rk)
                tl.store(A_ptr + j * stride_am + l * stride_an, A_value)

    # Compute Q^H * b
    for i in range(k):
        for j in range(n):
            QHb_value = 0.0
            for l in range(m):
                QHb_value += tl.load(A_ptr + l * stride_am + j * stride_an) * tl.load(b_ptr + l * stride_bk + i * stride_bn)
            tl.store(Q_ptr + j * stride_qn + i * stride_qk, QHb_value)

    # Solve R * x = Q^H * b
    for i in range(n - 1, -1, -1):
        for j in range(k):
            x_value = tl.load(Q_ptr + i * stride_qn + j * stride_qk)
            for l in range(i + 1, n):
                x_value -= tl.load(R_ptr + i * stride_rn + l * stride_rk) * tl.load(x_ptr + l * stride_xn + j * stride_xk)
            x_value /= tl.load(R_ptr + i * stride_rn + i * stride_rk)
            tl.store(x_ptr + i * stride_xn + j * stride_xk, x_value)

import torch
import triton
import triton.language as tl

def least_squares_qr(A, b, *, mode='reduced', out=None) -> torch.Tensor:
    """
    Solves the least squares problem for an overdetermined system of linear equations using QR decomposition.
    
    Parameters:
    - A (Tensor): Coefficient matrix of shape (*, m, n), where * is zero or more batch dimensions.
    - b (Tensor): Right-hand side vector or matrix of shape (*, m) or (*, m, k), where k is the number of right-hand sides.
    - mode (str, optional): Determines the type of QR decomposition to use. One of 'reduced' (default) or 'complete'.
    - out (Tensor, optional): Output tensor. Ignored if None. Default: None.
    
    Returns:
    - Tensor: The least squares solution x that minimizes the Euclidean 2-norm |Ax - b|_2.
    """
    *batch, m, n = A.shape
    *batch, m, k = b.shape if b.dim() == 3 else (*batch, m, 1)
    
    # Ensure A and b have the same batch dimensions
    if len(batch) == 0:
        A = A.unsqueeze(0)
        b = b.unsqueeze(0)
    
    # Allocate output tensor
    if out is None:
        out = torch.empty((*batch, n, k), device=A.device, dtype=A.dtype)
    
    # Flatten batch dimensions
    A = A.view(-1, m, n)
    b = b.view(-1, m, k)
    out = out.view(-1, n, k)
    
    # Compute QR decomposition
    Q, R = torch.linalg.qr(A, mode=mode)
    
    # Compute Q^H * b
    QHb = torch.matmul(Q.transpose(-2, -1), b)
    
    # Solve R * x = Q^H * b
    x = torch.linalg.solve(R, QHb)
    
    # Copy result to output tensor
    out.copy_(x)
    
    # Restore batch dimensions
    out = out.view(*batch, n, k)
    
    return out
