import torch
import triton
import triton.language as tl

@triton.jit
def matvec_kernel(A_ptr, x_ptr, y_ptr, alpha, beta, n, m, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)
    
    # Load the row of A
    A_row = tl.load(A_ptr + row_idx * m + col_idx, mask=col_idx < m, other=0.0)
    # Load the vector x
    x_vec = tl.load(x_ptr + col_idx, mask=col_idx < m, other=0.0)
    
    # Compute dot product for the row
    dot_product = tl.sum(A_row * x_vec, axis=0)
    
    # Scale by alpha
    dot_product *= alpha
    
    # Load y and scale by beta
    y_val = tl.load(y_ptr + row_idx)
    y_val = beta * y_val + dot_product
    
    # Store the result back to y
    tl.store(y_ptr + row_idx, y_val)

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, m = A.shape
    BLOCK_SIZE = 1024  # You may adjust this based on your hardware capabilities
    
    # Launch the Triton kernel
    grid = lambda META: (n, )
    matvec_kernel[grid](A, x, y, alpha, beta, n, m, BLOCK_SIZE)
    
    # Compute the dot product of the updated y with x
    result = torch.dot(y, x)
    
    return result
