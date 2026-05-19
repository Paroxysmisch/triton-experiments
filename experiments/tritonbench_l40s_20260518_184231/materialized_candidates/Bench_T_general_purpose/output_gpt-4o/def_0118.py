import torch
import triton
import triton.language as tl

@triton.jit
def triton_solve_and_add_scaled_vector(A_ptr, b_ptr, y_ptr, x_ptr, alpha, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load A and b
    A = tl.load(A_ptr + offs[:, None] * n + offs[None, :], mask=offs[:, None] < n)
    b = tl.load(b_ptr + offs, mask=offs < n)
    
    # Solve Ax = b using forward substitution
    x = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        if offs[i] < n:
            sum_ = tl.dot(A[i, :i], x[:i])
            x[i] = (b[i] - sum_) / A[i, i]
    
    # Load y and compute alpha * y
    y = tl.load(y_ptr + offs, mask=offs < n)
    scaled_y = alpha * y
    
    # Add the scaled vector to the solution
    x += scaled_y
    
    # Store the result
    tl.store(x_ptr + offs, x, mask=offs < n)

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    assert A.shape[0] == A.shape[1], "Matrix A must be square"
    assert b.shape[0] == A.shape[0], "Vector b must have the same number of rows as A"
    assert y.shape[0] == A.shape[0], "Vector y must have the same number of rows as A"
    
    n = A.shape[0]
    BLOCK_SIZE = 128  # Adjust this based on your GPU's capabilities
    
    # Allocate memory for the result
    x = torch.empty_like(b, dtype=torch.float32)
    
    # Launch the Triton kernel
    triton_solve_and_add_scaled_vector[(n + BLOCK_SIZE - 1) // BLOCK_SIZE](
        A_ptr=A.data_ptr(),
        b_ptr=b.data_ptr(),
        y_ptr=y.data_ptr(),
        x_ptr=x.data_ptr(),
        alpha=alpha,
        n=n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return x
