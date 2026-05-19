import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiply_symmetric_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, n, m, p):
    # Define the grid size
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Allocate shared memory for intermediate results
    C_temp = tl.zeros((n, p), dtype=tl.float32)

    # Load A and B into shared memory
    if row < n and col < p:
        # First operation: C = alpha * torch.mm(A, B) + beta * C
        for k in range(m):
            C_temp[row, col] += tl.load(A_ptr + row * m + k) * tl.load(B_ptr + k * p + col)
        C_temp[row, col] = alpha * C_temp[row, col] + beta * tl.load(C_ptr + row * p + col)

        # Update C with the symmetric operation: C = alpha * torch.mm(C, C.T) + beta * C
        for k in range(n):
            C_temp[row, col] += tl.load(C_ptr + row * p + k) * tl.load(C_ptr + col * p + k)
        tl.store(C_ptr + row * p + col, alpha * C_temp[row, col] + beta * tl.load(C_ptr + row * p + col))

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, m = A.shape
    _, p = B.shape

    # Launch the Triton kernel
    matrix_multiply_symmetric_kernel[(n, p)](A.data_ptr(), B.data_ptr(), C.data_ptr(), alpha, beta, n, m, p)

    return C
