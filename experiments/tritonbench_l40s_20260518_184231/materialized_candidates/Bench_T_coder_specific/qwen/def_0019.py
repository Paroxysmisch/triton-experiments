import triton
from triton.language import *

@triton.jit
def lu_decompose(A_ptr, P_ptr, L_ptr, U_ptr, n):
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(n, BLOCK_SIZE)
    
    # Initialize P as identity matrix
    p_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(P_ptr + p_idx * n + p_idx, 1.0, mask=p_idx < n)
    
    # Perform LU decomposition
    for k in range(BLOCK_SIZE):
        # Pivot row selection
        max_abs_val = 0.0
        pivot_row = -1
        for i in range(k, n):
            abs_val = abs(tl.load(A_ptr + i * n + k))
            if abs_val > max_abs_val:
                max_abs_val = abs_val
                pivot_row = i
        
        # Swap rows if necessary
        if pivot_row != k:
            # Update P
            temp = tl.load(P_ptr + k * n + pivot_row)
            tl.store(P_ptr + k * n + pivot_row, 1.0)
            tl.store(P_ptr + pivot_row * n + k, temp)
        
        # Compute L[k, k]
        lkk = 1.0 / tl.load(A_ptr + k * n + k)
        tl.store(L_ptr + k * n + k, lkk)
        
        # Compute L[i, k] and update A
        for i in range(k + 1, n):
            lik = tl.load(A_ptr + i * n + k) * lkk
            tl.store(L_ptr + i * n + k, lik)
            for j in range(k, n):
                ajj = tl.load(A_ptr + k * n + j)
                ai_j = tl.load(A_ptr + i * n + j) - lik * ajj
                tl.store(A_ptr + i * n + j, ai_j)

@triton.jit
def forward(A_ptr, b_ptr, x_ptr, P_ptr, L_ptr, U_ptr, n):
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(n, BLOCK_SIZE)
    
    # Solve Ly = Pb
    y = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(n):
        y_i = tl.load(b_ptr + i)
        for j in range(i):
            y_i -= tl.load(L_ptr + i * n + j) * y[j]
        y_i /= tl.load(L_ptr + i * n + i)
        y[i] = y_i
    
    # Apply permutation P to y
    for i in range(BLOCK_SIZE):
        permuted_y = y[pid]
        tl.store(x_ptr + pid, permuted_y)
    
    # Solve Ux = y
    for i in reversed(range(n)):
        xi = y[i]
        for j in range(i + 1, n):
            xi -= tl.load(U_ptr + i * n + j) * x[j]
        xi /= tl.load(U_ptr + i * n + i)
        tl.store(x_ptr + pid, xi)

# Wrapper function
def fused_lu_solve(A: Tensor, b: Tensor) -> Tensor:
    n = A.shape[0]
    assert A.shape == (n, n), "Matrix A must be square"
    assert b.shape == (n,), "Vector b must have the same number of elements as the number of rows in A"
    
    A_device = triton.empty_like(A)
    P_device = triton.empty((n, n), dtype=A.dtype)
    L_device = triton.empty((n, n), dtype=A.dtype)
    U_device = triton.empty((n, n), dtype=A.dtype)
    x_device = triton.empty_like(b)
    
    # Copy A to device
    A_device.copy_to_device(A)
    
    # Perform LU decomposition
    lu_decompose[A_device.numel()](A_device.data_ptr(), P_device.data_ptr(), L_device.data_ptr(), U_device.data_ptr(), n)
    
    # Forward pass to solve Ax = b
    forward[A_device.numel()](A_device.data_ptr(), b.data_ptr(), x_device.data_ptr(), P_device.data_ptr(), L_device.data_ptr(), U_device.data_ptr(), n)
    
    return x_device.to_host()
