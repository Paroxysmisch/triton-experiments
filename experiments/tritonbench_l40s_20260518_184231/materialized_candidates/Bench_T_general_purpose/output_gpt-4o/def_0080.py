import torch
import triton
import triton.language as tl

@triton.jit
def qr_decomposition(A_ptr, Q_ptr, R_ptr, m, n, stride_am, stride_an, stride_qm, stride_qn, stride_rm, stride_rn, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, BLOCK_SIZE)
    
    A = tl.load(A_ptr + row[:, None] * stride_am + col[None, :] * stride_an, mask=(row[:, None] < m) & (col[None, :] < n), other=0.0)
    
    # QR decomposition using Gram-Schmidt process
    Q = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    R = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    
    for k in range(n):
        if k < BLOCK_SIZE:
            r_kk = tl.sqrt(tl.sum(A[:, k] ** 2))
            R[k, k] = r_kk
            Q[:, k] = A[:, k] / r_kk
            
            for j in range(k + 1, n):
                r_kj = tl.sum(Q[:, k] * A[:, j])
                R[k, j] = r_kj
                A[:, j] = A[:, j] - r_kj * Q[:, k]
    
    # Store results back
    tl.store(Q_ptr + row[:, None] * stride_qm + col[None, :] * stride_qn, Q, mask=(row[:, None] < m) & (col[None, :] < n))
    tl.store(R_ptr + row[:, None] * stride_rm + col[None, :] * stride_rn, R, mask=(row[:, None] < n) & (col[None, :] < n))

@triton.jit
def solve_upper_triangular(R_ptr, y_ptr, x_ptr, n, stride_rm, stride_rn, stride_y, stride_x, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    col = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    R = tl.load(R_ptr + col[None, :] * stride_rm + col[:, None] * stride_rn, mask=(col[None, :] < n) & (col[:, None] < n), other=0.0)
    y = tl.load(y_ptr + col * stride_y, mask=(col < n), other=0.0)
    
    x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(n-1, -1, -1):
        if i < BLOCK_SIZE:
            x[i] = (y[i] - tl.sum(R[i, i+1:] * x[i+1:])) / R[i, i]
    
    tl.store(x_ptr + col * stride_x, x, mask=(col < n))

def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    k = b.shape[1]
    
    # Allocate Q, R, and intermediate results
    Q = torch.empty((m, n), dtype=A.dtype, device=A.device)
    R = torch.empty((n, n), dtype=A.dtype, device=A.device)
    Qt_b = torch.empty((n, k), dtype=A.dtype, device=A.device)
    x = torch.empty((n, k), dtype=A.dtype, device=A.device)
    
    # Launch Triton kernels
    qr_decomposition[(1,)](A, Q, R, m, n, A.stride(0), A.stride(1), Q.stride(0), Q.stride(1), R.stride(0), R.stride(1), BLOCK_SIZE=128)
    
    # Compute Q^T b
    Qt_b = torch.matmul(Q.T, b)
    
    # Solve Rx = Q^T b
    solve_upper_triangular[(1,)](R, Qt_b, x, n, R.stride(0), R.stride(1), Qt_b.stride(0), x.stride(0), BLOCK_SIZE=128)
    
    return x
