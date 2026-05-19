import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(
    A_ptr, L_ptr,
    stride_am, stride_ak,
    stride_lm, stride_lk,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Row and column indices
    row = pid // (n // BLOCK_SIZE)
    col = pid % (n // BLOCK_SIZE)
    
    # Load block
    offs_am = row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_ak = col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(A_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    
    # Only process lower triangular part
    if row >= col:
        if row == col:
            # Diagonal block - Cholesky decomposition
            for k in range(BLOCK_SIZE):
                # Update diagonal element
                a[k, k] = tl.sqrt(a[k, k])
                
                # Update column below diagonal
                if k < BLOCK_SIZE - 1:
                    a[k+1:, k] = a[k+1:, k] / a[k, k]
                    
                # Update remaining elements
                if k < BLOCK_SIZE - 1:
                    a[k+1:, k+1:] -= tl.outer(a[k+1:, k], a[k+1:, k])
        
        # Store result in L
        tl.store(L_ptr + offs_am[:, None] * stride_lm + offs_ak[None, :] * stride_lk, a)

@triton.jit
def forward_substitution_kernel(
    L_ptr, b_ptr, y_ptr,
    stride_lm, stride_lk,
    stride_bm, stride_bk,
    stride_ym, stride_yk,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load b
    b = tl.load(b_ptr + row[:, None] * stride_bm + tl.arange(0, k)[None, :] * stride_bk)
    y = b.copy()
    
    # Forward substitution
    for i in range(n):
        if i < row[0]:
            l = tl.load(L_ptr + row[:, None] * stride_lm + i * stride_lk)
            yi = tl.load(y_ptr + i * stride_ym + tl.arange(0, k) * stride_yk)
            y -= tl.outer(l, yi)
        elif i <= row[-1]:
            mask = row >= i
            l = tl.load(L_ptr + row[:, None] * stride_lm + i * stride_lk)
            y = tl.where(mask[:, None], y / l[:, None], y)
    
    # Store result
    tl.store(y_ptr + row[:, None] * stride_ym + tl.arange(0, k)[None, :] * stride_yk, y)

@triton.jit
def backward_substitution_kernel(
    L_ptr, y_ptr, x_ptr,
    stride_lm, stride_lk,
    stride_ym, stride_yk,
    stride_xm, stride_xk,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = (n - 1) - (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    
    # Load y
    y = tl.load(y_ptr + row[:, None] * stride_ym + tl.arange(0, k)[None, :] * stride_yk)
    x = y.copy()
    
    # Backward substitution
    for i in range(n-1, -1, -1):
        if i > row[-1]:
            l = tl.load(L_ptr + i * stride_lm + row * stride_lk)
            xi = tl.load(x_ptr + i * stride_xm + tl.arange(0, k) * stride_xk)
            x -= tl.outer(l, xi)
        elif i >= row[0]:
            mask = row <= i
            l = tl.load(L_ptr + i * stride_lm + row * stride_lk)
            x = tl.where(mask[:, None], x / l[:, None], x)
    
    # Store result
    tl.store(x_ptr + row[:, None] * stride_xm + tl.arange(0, k)[None, :] * stride_xk, x)

def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solves the system Ax = b using Cholesky decomposition.
    
    Args:
        A: Symmetric positive-definite matrix of shape (n, n)
        b: Right-hand side tensor of shape (n, k)
    
    Returns:
        x: Solution tensor of shape (n, k)
    """
    assert A.dim() == 2 and A.size(0) == A.size(1), "A must be a square matrix"
    assert b.dim() == 2 and b.size(0) == A.size(0), "b must have compatible dimensions with A"
    
    n = A.size(0)
    k = b.size(1)
    device = A.device
    
    # Initialize L matrix for Cholesky decomposition
    L = torch.zeros_like(A, device=device)
    
    # Compute Cholesky decomposition
    BLOCK_SIZE = 32
    grid = lambda meta: (triton.cdiv(n, BLOCK_SIZE) * triton.cdiv(n, BLOCK_SIZE),)
    cholesky_kernel[grid](
        A_ptr=A, L_ptr=L,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_lm=L.stride(0), stride_lk=L.stride(1),
        n=n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Initialize y and x for solving Ly = b and L^T x = y
    y = torch.zeros_like(b, device=device)
    x = torch.zeros_like(b, device=device)
    
    # Forward substitution (Ly = b)
    grid = lambda meta: (triton.cdiv(n, BLOCK_SIZE),)
    forward_substitution_kernel[grid](
        L_ptr=L, b_ptr=b, y_ptr=y,
        stride_lm=L.stride(0), stride_lk=L.stride(1),
        stride_bm=b.stride(0), stride_bk=b.stride(1),
        stride_ym=y.stride(0), stride_yk=y.stride(1),
        n=n, k=k, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Backward substitution (L^T x = y)
    backward_substitution_kernel[grid](
        L_ptr=L.T.contiguous(), y_ptr=y, x_ptr=x,
        stride_lm=L.T.stride(0), stride_lk=L.T.stride(1),
        stride_ym=y.stride(0), stride_yk=y.stride(1),
        stride_xm=x.stride(0), stride_xk=x.stride(1),
        n=n, k=k, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return x
