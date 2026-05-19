import torch
import triton
import triton.language as tl

@triton.jit
def matrix_vector_kernel(A_ptr, x_ptr, y_ptr, alpha, beta, n, m):
    """
    Computes the matrix-vector product and updates y in-place.

    Args:
    A: input matrix of shape (n, m)
    x: input vector of shape (m,)
    y: target vector of shape (n,)
    alpha: scalar multiplier for A @ x
    beta: scalar multiplier for y
    """
    row = tl.program_id(0)
    if row < n:
        # Load the row of A and the vector x
        A_row = tl.load(A_ptr + row * m + tl.arange(0, m))
        x = tl.load(x_ptr)
        
        # Compute the matrix-vector product
        y_update = alpha * tl.dot(A_row, x) + beta * tl.load(y_ptr + row)
        
        # Store the updated value back to y
        tl.store(y_ptr + row, y_update)

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, m = A.shape
    assert x.shape == (m,)
    assert y.shape == (n,)
    
    # Launch the kernel
    grid = (n,)
    matrix_vector_kernel[grid](A, x, y, alpha, beta, n, m)
    
    # Calculate the dot product
    result = torch.dot(y, x)
    return result
