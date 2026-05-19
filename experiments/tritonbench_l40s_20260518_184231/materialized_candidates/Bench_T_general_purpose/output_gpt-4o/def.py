import torch
import triton
import triton.language as tl

@triton.jit
def lu_decomposition(A, P, L, U, n, pivot):
    # Triton kernel for LU decomposition with optional pivoting
    pid = tl.program_id(0)
    row = tl.arange(0, n)
    col = tl.arange(0, n)
    
    # Load the current row of A
    a_row = tl.load(A + pid * n * n + row * n + col)
    
    # Initialize L and U
    tl.store(L + pid * n * n + row * n + col, tl.where(row == col, 1.0, 0.0))
    tl.store(U + pid * n * n + row * n + col, a_row)
    
    # Perform LU decomposition
    for i in range(n):
        if pivot:
            # Find pivot
            max_idx = tl.argmax(tl.abs(a_row[i:]))
            # Swap rows in P, L, and U
            tl.atomic_xchg(P + pid * n + i, max_idx + i)
            tl.atomic_xchg(U + pid * n * n + i * n + col, U + pid * n * n + max_idx * n + col)
            tl.atomic_xchg(L + pid * n * n + i * n + col, L + pid * n * n + max_idx * n + col)
        
        # Compute multipliers and eliminate below pivot
        for j in range(i + 1, n):
            multiplier = U[j, i] / U[i, i]
            tl.store(L + pid * n * n + j * n + i, multiplier)
            U[j, i:] -= multiplier * U[i, i:]

@triton.jit
def forward_substitution(L, B, Y, n, k):
    # Triton kernel for forward substitution
    pid = tl.program_id(0)
    row = tl.arange(0, n)
    col = tl.arange(0, k)
    
    for i in range(n):
        y = tl.load(B + pid * n * k + i * k + col)
        for j in range(i):
            y -= tl.load(L + pid * n * n + i * n + j) * tl.load(Y + pid * n * k + j * k + col)
        y /= tl.load(L + pid * n * n + i * n + i)
        tl.store(Y + pid * n * k + i * k + col, y)

@triton.jit
def backward_substitution(U, Y, X, n, k):
    # Triton kernel for backward substitution
    pid = tl.program_id(0)
    row = tl.arange(0, n)
    col = tl.arange(0, k)
    
    for i in range(n - 1, -1, -1):
        x = tl.load(Y + pid * n * k + i * k + col)
        for j in range(i + 1, n):
            x -= tl.load(U + pid * n * n + i * n + j) * tl.load(X + pid * n * k + j * k + col)
        x /= tl.load(U + pid * n * n + i * n + i)
        tl.store(X + pid * n * k + i * k + col, x)

def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
    n = A.shape[-1]
    k = Bs.shape[-1]
    batch_size = A.shape[0] if A.ndim > 2 else 1
    
    # Allocate space for L, U, P
    L = torch.zeros_like(A)
    U = torch.zeros_like(A)
    P = torch.arange(n, device=A.device).expand(batch_size, n)
    
    # Perform LU decomposition
    lu_decomposition[(batch_size,)](A, P, L, U, n, pivot)
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(Bs)
    
    # Allocate space for intermediate Y
    Y = torch.empty_like(Bs)
    
    # Forward substitution
    forward_substitution[(batch_size,)](L, Bs, Y, n, k)
    
    # Backward substitution
    backward_substitution[(batch_size,)](U, Y, out, n, k)
    
    return out
