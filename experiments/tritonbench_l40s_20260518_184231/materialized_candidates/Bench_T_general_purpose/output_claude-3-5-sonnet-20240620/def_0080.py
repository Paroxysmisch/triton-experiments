import torch
import triton
import triton.language as tl

@triton.jit
def qr_solve_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, q_ptr, r_ptr, x_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_an,
    stride_bm, stride_bk,
    stride_qm, stride_qn,
    stride_rm, stride_rn,
    stride_xn, stride_xk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # ---------------- Matrix multiplication Q^T * b ----------------
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Load Q^T and b
    q = tl.load(q_ptr + offs_am[:, None] * stride_qm + tl.arange(0, M)[None, :] * stride_qn)
    b = tl.load(b_ptr + tl.arange(0, M)[:, None] * stride_bm + offs_bn[None, :] * stride_bk)
    
    # Compute Q^T * b
    qtb = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, M, BLOCK_SIZE_M):
        qtb += tl.dot(q, b)
    
    # ---------------- Solve R * x = Q^T * b ----------------
    # Load R
    r = tl.load(r_ptr + offs_am[:, None] * stride_rm + tl.arange(0, N)[None, :] * stride_rn)
    
    # Back substitution
    x = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for i in range(N-1, -1, -1):
        qtb[i, :] -= tl.sum(r[i, i+1:N] * x[i+1:N, :], axis=0)
        x[i, :] = qtb[i, :] / r[i, i]
    
    # Write output
    tl.store(x_ptr + offs_am[:, None] * stride_xn + offs_bn[None, :] * stride_xk, x)

def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solves the linear system Ax = b using QR decomposition.
    
    Args:
        A: Input matrix of shape (m, n) where m >= n
        b: Right-hand side tensor of shape (m, k)
    
    Returns:
        x: Solution tensor of shape (n, k)
    """
    assert A.dim() == 2, "A must be a 2D tensor"
    assert b.dim() == 2, "b must be a 2D tensor"
    
    m, n = A.shape
    assert m >= n, "Number of rows must be greater than or equal to number of columns"
    assert b.shape[0] == m, "Incompatible dimensions between A and b"
    
    # Compute QR decomposition
    Q, R = torch.linalg.qr(A)
    
    # Initialize output tensor
    k = b.shape[1]
    x = torch.empty((n, k), device=A.device, dtype=A.dtype)
    
    # Launch Triton kernel
    grid = lambda meta: (
        triton.cdiv(n, meta['BLOCK_SIZE_M']) * triton.cdiv(k, meta['BLOCK_SIZE_N']),
    )
    
    qr_solve_kernel[grid](
        A.contiguous().data_ptr(),
        b.contiguous().data_ptr(),
        Q.contiguous().data_ptr(),
        R.contiguous().data_ptr(),
        x.data_ptr(),
        m, n, k,
        A.stride(0), A.stride(1),
        b.stride(0), b.stride(1),
        Q.stride(0), Q.stride(1),
        R.stride(0), R.stride(1),
        x.stride(0), x.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
    )
    
    return x
