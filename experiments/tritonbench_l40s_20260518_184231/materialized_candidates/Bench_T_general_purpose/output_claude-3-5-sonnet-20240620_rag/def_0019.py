import torch
import triton
import triton.language as tl

@triton.jit
def lu_decomposition_kernel(
    A,  # Pointer to input matrix (n x n)
    L,  # Pointer to lower triangular matrix
    U,  # Pointer to upper triangular matrix
    P,  # Pointer to permutation matrix
    n,  # Matrix dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate row and column indices
    row_idx = pid // (n // BLOCK_SIZE)
    col_idx = pid % (n // BLOCK_SIZE)
    
    # Load block offsets
    row_start = row_idx * BLOCK_SIZE
    col_start = col_idx * BLOCK_SIZE
    
    # Initialize block ranges
    rows = row_start + tl.arange(0, BLOCK_SIZE)
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    
    # Load matrix block
    mask = (rows[:, None] < n) & (cols[None, :] < n)
    a_block = tl.load(A + rows[:, None] * n + cols[None, :], mask=mask, other=0.0)
    
    # Perform block LU decomposition
    for k in range(BLOCK_SIZE):
        if k < min(rows.shape[0], cols.shape[0]):
            # Find pivot
            pivot_row = k + tl.argmax(tl.abs(a_block[k:, k]), axis=0)
            
            # Swap rows if necessary
            if pivot_row != k:
                temp = a_block[k].to(tl.float32)
                a_block = tl.where(
                    tl.arange(0, BLOCK_SIZE) == k,
                    a_block[pivot_row],
                    a_block
                )
                a_block = tl.where(
                    tl.arange(0, BLOCK_SIZE) == pivot_row,
                    temp,
                    a_block
                )
            
            # Update L and U blocks
            diag = a_block[k, k]
            if diag != 0:
                l_col = a_block[k+1:, k] / diag
                u_row = a_block[k, k+1:]
                
                # Update remaining block
                a_block[k+1:, k+1:] -= tl.outer(l_col, u_row)
    
    # Store results
    tl.store(L + rows[:, None] * n + cols[None, :], tl.tril(a_block, -1) + tl.eye(BLOCK_SIZE), mask=mask)
    tl.store(U + rows[:, None] * n + cols[None, :], tl.triu(a_block), mask=mask)

@triton.jit
def solve_triangular_kernel(
    L,      # Lower triangular matrix
    U,      # Upper triangular matrix
    b,      # Right-hand side vector
    x,      # Solution vector
    n,      # Matrix dimension
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row < n
    
    # Forward substitution (Ly = b)
    y = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, n, BLOCK_SIZE):
        l_block = tl.load(L + row[:, None] * n + (i + tl.arange(0, BLOCK_SIZE))[None, :], 
                         mask=mask[:, None], other=0.0)
        b_block = tl.load(b + i + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        y = y + tl.sum(l_block * b_block[None, :], axis=1)
    
    # Backward substitution (Ux = y)
    x_temp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(n-1, -1, -BLOCK_SIZE):
        u_block = tl.load(U + row[:, None] * n + (i + tl.arange(0, BLOCK_SIZE))[None, :],
                         mask=mask[:, None], other=0.0)
        x_temp = x_temp + tl.sum(u_block * y[None, :], axis=1)
    
    # Store result
    tl.store(x + row, x_temp, mask=mask)

def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solves the linear system Ax = b using LU decomposition.
    
    Args:
        A: Input matrix of shape (n, n)
        b: Right-hand side vector of shape (n,)
        
    Returns:
        x: Solution vector of shape (n,)
    """
    assert A.dim() == 2 and A.size(0) == A.size(1), "Matrix A must be square"
    assert b.dim() == 1 and b.size(0) == A.size(0), "Dimensions of A and b must match"
    
    n = A.size(0)
    device = A.device
    
    # Initialize matrices
    L = torch.zeros_like(A)
    U = torch.zeros_like(A)
    P = torch.eye(n, device=device)
    x = torch.zeros_like(b)
    
    # Determine block size (power of 2 for better performance)
    BLOCK_SIZE = min(128, triton.next_power_of_2(n))
    
    # Launch LU decomposition kernel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
    lu_decomposition_kernel[grid](
        A, L, U, P, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Launch solve kernel
    solve_triangular_kernel[grid](
        L, U, b, x, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return x
