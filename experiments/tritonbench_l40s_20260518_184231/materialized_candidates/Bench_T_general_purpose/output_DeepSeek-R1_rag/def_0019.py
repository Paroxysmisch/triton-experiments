import torch
import triton
import triton.language as tl

@triton.jit
def forward_substitution_kernel(
    L_ptr,  # Pointer to L matrix (lower triangular, 1s on diagonal)
    b_ptr,  # Pointer to permuted vector Pb
    y_ptr,  # Pointer to output vector y
    n: int,  # Size of the matrix (n x n)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == 0:
        for i in range(n):
            # Initialize sum for the current row i
            sum = 0.0
            # Accumulate L[i, j] * y[j] for j < i
            for j in range(i):
                # Calculate the offset for L[i, j]
                offset_L = i * n + j
                L_ij = tl.load(L_ptr + offset_L)
                # Load y[j]
                y_j = tl.load(y_ptr + j)
                sum += L_ij * y_j
            # Load Pb[i]
            b_i = tl.load(b_ptr + i)
            # Compute y[i] = Pb[i] - sum (since L[i,i] = 1)
            y_i = b_i - sum
            # Store y[i]
            tl.store(y_ptr + i, y_i)

@triton.jit
def backward_substitution_kernel(
    U_ptr,  # Pointer to U matrix (upper triangular)
    y_ptr,  # Pointer to vector y from forward substitution
    x_ptr,  # Pointer to output vector x
    n: int,  # Size of the matrix (n x n)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == 0:
        # Iterate from the last row to the first
        for i in range(n - 1, -1, -1):
            sum = 0.0
            # Accumulate U[i, j] * x[j] for j > i
            for j in range(i + 1, n):
                # Calculate the offset for U[i, j]
                offset_U = i * n + j
                U_ij = tl.load(U_ptr + offset_U)
                # Load x[j]
                x_j = tl.load(x_ptr + j)
                sum += U_ij * x_j
            # Load y[i]
            y_i = tl.load(y_ptr + i)
            # Load U[i, i] to handle division
            U_ii = tl.load(U_ptr + i * n + i)
            # Compute x[i] = (y_i - sum) / U[i,i]
            x_i = (y_i - sum) / U_ii
            # Store x[i]
            tl.store(x_ptr + i, x_i)

def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Ensure A is square and b has compatible shape
    assert A.dim() == 2 and A.size(0) == A.size(1), "A must be a square matrix"
    n = A.size(0)
    assert b.size() == (n,), "b must be a vector of size n"
    
    # Perform LU decomposition on A
    A_LU, pivots = torch.linalg.lu_factor(A)
    L, U = torch.lu_unpack(A_LU, pivots)
    
    # Apply permutation to b (adjust pivots to 0-based index)
    pivots_0based = pivots - 1
    Pb = b[pivots_0based]
    
    # Allocate tensors for intermediate results
    y = torch.zeros_like(Pb)
    x = torch.zeros_like(y)
    
    # Define the grid with a single block to enforce sequential execution
    grid = lambda _: (1,)
    
    # Perform forward substitution: solve Ly = Pb
    forward_substitution_kernel[grid](L, Pb, y, n, BLOCK_SIZE=1)
    
    # Perform backward substitution: solve Ux = y
    backward_substitution_kernel[grid](U, y, x, n, BLOCK_SIZE=1)
    
    return x
