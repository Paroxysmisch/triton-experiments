import triton
import triton.language as tl

@triton.jit
def matrix_multiply_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, n, m, p):
    # Define the grid size
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Initialize the accumulator
    acc = 0.0

    # Compute the matrix product
    for k in range(m):
        acc += tl.load(A_ptr + row * m + k) * tl.load(B_ptr + k * p + col)

    # Scale and add to C
    C_value = alpha * acc + beta * tl.load(C_ptr + row * p + col)
    tl.store(C_ptr + row * p + col, C_value)

import torch

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    # Ensure C has at least two rows
    assert C.shape[0] >= 2, "C must have at least two rows for dot product."

    # Get dimensions
    n, m = A.shape
    p = B.shape[1]

    # Allocate output tensor
    C_out = torch.empty_like(C)

    # Launch the Triton kernel
    grid = (n, p)
    matrix_multiply_kernel[grid](A.data_ptr(), B.data_ptr(), C_out.data_ptr(), alpha, beta, n, m, p)

    # Compute the dot product of the first two rows
    result = torch.dot(C_out[0], C_out[1])
    
    return result
