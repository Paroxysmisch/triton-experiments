import torch
import triton
import triton.language as tl
import math

@triton.jit
def cholesky_kernel(A_ptr, L_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Iterate over each column to compute L
    for j in range(n):
        # Compute diagonal element L[j, j]
        sum_sq = 0.0
        for k in range(j):
            L_jk = tl.load(L_ptr + j * n + k)
            sum_sq += L_jk * L_jk
        A_jj = tl.load(A_ptr + j * n + j)
        L_jj = tl.sqrt(A_jj - sum_sq)
        tl.store(L_ptr + j * n + j, L_jj)
        
        # Compute elements below the diagonal in column j
        for i in range(j + 1, n):
            sum_val = 0.0
            for k in range(j):
                L_ik = tl.load(L_ptr + i * n + k)
                L_jk = tl.load(L_ptr + j * n + k)
                sum_val += L_ik * L_jk
            A_ij = tl.load(A_ptr + i * n + j)
            L_ij = (A_ij - sum_val) / L_jj
            tl.store(L_ptr + i * n + j, L_ij)

@triton.jit
def forward_substitute_kernel(L_ptr, b_ptr, y_ptr, n, k, BLOCK_SIZE: tl.constexpr):
    col = tl.program_id(0)
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            L_ij = tl.load(L_ptr + i * n + j)
            y_j = tl.load(y_ptr + j * k + col)
            sum_val += L_ij * y_j
        b_i = tl.load(b_ptr + i * k + col)
        L_ii = tl.load(L_ptr + i * n + i)
        y_i = (b_i - sum_val) / L_ii
        tl.store(y_ptr + i * k + col, y_i)

@triton.jit
def backward_substitute_kernel(L_ptr, y_ptr, x_ptr, n, k, BLOCK_SIZE: tl.constexpr):
    col = tl.program_id(0)
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            L_ji = tl.load(L_ptr + j * n + i)
            x_j = tl.load(x_ptr + j * k + col)
            sum_val += L_ji * x_j
        y_i = tl.load(y_ptr + i * k + col)
        L_ii = tl.load(L_ptr + i * n + i)
        x_i = (y_i - sum_val) / L_ii
        tl.store(x_ptr + i * k + col, x_i)

def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = A.size(0)
    k = b.size(1)
    assert A.size(1) == n and b.size(0) == n, "Invalid input dimensions"
    
    # Allocate output tensors
    L = torch.zeros_like(A)
    y = torch.zeros_like(b)
    x = torch.zeros_like(b)
    
    # Launch Cholesky decomposition kernel
    cholesky_kernel[(1,)](A, L, n, BLOCK_SIZE=n)
    
    # Launch forward substitution kernel
    forward_substitute_kernel[(k,)](L, b, y, n, k, BLOCK_SIZE=1)
    
    # Launch backward substitution kernel
    backward_substitute_kernel[(k,)](L, y, x, n, k, BLOCK_SIZE=1)
    
    return x
