import triton
import triton.language as tl
import torch

# Triton kernel for LU decomposition
@triton.jit
def lu_decomposition_kernel(
    A_ptr,  # Pointer to the input matrix A
    L_ptr,  # Pointer to the output lower triangular matrix L
    U_ptr,  # Pointer to the output upper triangular matrix U
    P_ptr,  # Pointer to the permutation matrix P
    N,  # Number of rows/columns in A
    BLOCK_SIZE: tl.constexpr  # Block size for the decomposition
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_SIZE)
    
    # Load the current row of A
    a_row = tl.load(A_ptr + row * N + col, mask=col < N)
    
    # Initialize L and U
    l_row = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    u_row = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Perform LU decomposition with pivoting
    for k in range(N):
        # Find pivot
        if row == k:
            pivot = tl.argmax(tl.abs(a_row[k:N]))
            tl.store(P_ptr + row, pivot)
        
        # Broadcast pivot
        pivot = tl.load(P_ptr + row)
        pivot_row = tl.load(A_ptr + pivot * N + col, mask=col < N)
        
        # Swap rows
        a_row = tl.where(col < N, pivot_row, a_row)
        
        # Compute U
        u_row = tl.where(col >= k, a_row, u_row)
        
        # Compute L
        if row > k:
            l_row[k] = a_row[k] / u_row[k]
            a_row = a_row - l_row[k] * tl.load(U_ptr + k * N + col, mask=col < N)
        
        # Store L and U
        tl.store(L_ptr + row * N + col, l_row, mask=col < N)
        tl.store(U_ptr + row * N + col, u_row, mask=col < N)

# Triton kernel for forward and backward substitution
@triton.jit
def forward_backward_substitution_kernel(
    L_ptr,  # Pointer to the lower triangular matrix L
    U_ptr,  # Pointer to the upper triangular matrix U
    P_ptr,  # Pointer to the permutation matrix P
    B_ptr,  # Pointer to the right-hand side matrix B
    X_ptr,  # Pointer to the solution matrix X
    N,  # Number of rows/columns in A
    K,  # Number of right-hand sides
    BLOCK_SIZE: tl.constexpr  # Block size for substitution
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_SIZE)
    
    # Apply permutation to B
    permuted_b = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(K):
        pivot = tl.load(P_ptr + row)
        b_row = tl.load(B_ptr + pivot * K + col, mask=col < K)
        permuted_b = tl.where(col < K, b_row, permuted_b)
    
    # Forward substitution L * y = P^T * B
    y = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(N):
        l_row = tl.load(L_ptr + row * N + col, mask=col < N)
        y = tl.where(col < K, permuted_b - l_row * y, y)
        y = tl.where(col == k, y / l_row[k], y)
    
    # Backward substitution U * x = y
    x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(N-1, -1, -1):
        u_row = tl.load(U_ptr + row * N + col, mask=col < N)
        x = tl.where(col < K, y - u_row * x, x)
        x = tl.where(col == k, x / u_row[k], x)
    
    # Store the result
    tl.store(X_ptr + row * K + col, x, mask=col < K)

# Python wrapper function
def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
    assert A.ndim >= 2 and A.shape[-2] == A.shape[-1], "A must be a square matrix"
    assert Bs.ndim >= 2 and Bs.shape[-2] == A.shape[-1], "Bs must have compatible dimensions with A"
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(Bs)
    
    # Extract dimensions
    *batch_dims, n, _ = A.shape
    k = Bs.shape[-1]
    
    # Flatten batch dimensions
    batch_size = torch.prod(torch.tensor(batch_dims))
    A = A.view(batch_size, n, n)
    Bs = Bs.view(batch_size, n, k)
    out = out.view(batch_size, n, k)
    
    # Allocate L, U, and P matrices
    L = torch.zeros_like(A)
    U = torch.zeros_like(A)
    P = torch.zeros(batch_size, n, dtype=torch.int32, device=A.device)
    
    # Launch LU decomposition kernel
    grid = lambda meta: (batch_size, )
    lu_decomposition_kernel[grid](A, L, U, P, n, BLOCK_SIZE=32)
    
    # Launch forward and backward substitution kernel
    grid = lambda meta: (batch_size, )
    forward_backward_substitution_kernel[grid](L, U, P, Bs, out, n, k, BLOCK_SIZE=32)
    
    return out
