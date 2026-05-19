import triton
import triton.language as tl

@triton.jit
def solve_and_add_scaled_vector_kernel(
    A_ptr,  # Pointer to the upper triangular matrix A
    b_ptr,  # Pointer to the right-hand side vector or matrix b
    y_ptr,  # Pointer to the vector y
    x_ptr,  # Pointer to the output vector or matrix x
    alpha,  # Scaling factor for the vector y
    n,  # Size of the matrix A
    k,  # Number of columns in b (1 if b is a vector)
    BLOCK_SIZE: tl.constexpr
):
    # Compute the block ID and the thread ID within the block
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    
    # Compute the row and column indices for the current thread
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't access out-of-bounds memory
    mask = (row < n) & (col < k)
    
    # Load the elements of A, b, and y
    A = tl.load(A_ptr + row[:, None] * n + col[None, :], mask=mask, other=0.0)
    b = tl.load(b_ptr + row * k + col, mask=mask, other=0.0)
    y = tl.load(y_ptr + row, mask=row < n, other=0.0)
    
    # Solve the triangular system Ax = b
    x = tl.zeros((BLOCK_SIZE, k), dtype=tl.float32)
    for i in range(n - 1, -1, -1):
        x[i, :] = (b[i, :] - tl.sum(A[i, i+1:] * x[i+1:, :], axis=0)) / A[i, i]
    
    # Add the scaled vector y to the solution x
    x += alpha * y[:, None]
    
    # Store the result back to the output tensor
    tl.store(x_ptr + row * k + col, x, mask=mask)

import torch

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    # Ensure the input tensors are on the same device
    device = A.device
    A = A.to(device)
    b = b.to(device)
    y = y.to(device)
    
    # Solve the triangular system Ax = b
    x = torch.linalg.solve_triangular(A, b, upper=True)
    
    # Add the scaled vector alpha * y to the solution x
    x += alpha * y
    
    return x
