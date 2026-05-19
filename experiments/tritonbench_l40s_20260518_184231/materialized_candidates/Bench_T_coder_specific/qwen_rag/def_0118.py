import triton
import triton.language as tl
import torch

# Triton kernel to solve the triangular system Ax = b and add a scaled vector y to the solution x
@triton.jit
def solve_and_add_scaled_vector_kernel(
    A_ptr,  # Pointer to the input matrix A
    b_ptr,  # Pointer to the right-hand side vector or matrix b
    y_ptr,  # Pointer to the vector y
    x_ptr,  # Pointer to the output vector x
    n,      # Number of rows/columns in A, rows in b, and length of y
    k,      # Number of columns in b (if b is a matrix)
    alpha,  # Scaling factor for the vector y
    BLOCK_SIZE: tl.constexpr,  # Block size for the kernel
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    row = pid * BLOCK_SIZE + tid

    if row >= n:
        return

    # Load A[row, :]
    A_row = []
    for col in range(n):
        A_row.append(tl.load(A_ptr + row * n + col))

    # Load b[row, :]
    b_elem = tl.load(b_ptr + row)

    # Solve the triangular system Ax = b
    x_elem = b_elem
    for j in range(row):
        x_elem -= A_row[j] * x_elem

    # Store the solution x[row]
    tl.store(x_ptr + row, x_elem)

    # Load y[row]
    y_elem = tl.load(y_ptr + row)

    # Add the scaled vector y to the solution x
    final_x_elem = x_elem + alpha * y_elem

    # Store the final result
    tl.store(x_ptr + row, final_x_elem)

# Wrapper function to call the Triton kernel
def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    A = A.contiguous()
    b = b.contiguous()
    y = y.contiguous()

    assert A.ndim == 2, "A must be a 2D tensor"
    assert b.ndim in [1, 2], "b must be a 1D or 2D tensor"
    assert y.ndim == 1, "y must be a 1D tensor"

    n = A.shape[0]
    k = b.shape[1] if b.ndim == 2 else 1

    out = torch.zeros_like(b)

    BLOCK_SIZE = 32
    grid_size = (triton.cdiv(n, BLOCK_SIZE), triton.cdiv(k, BLOCK_SIZE))
    block_size = (BLOCK_SIZE, BLOCK_SIZE)

    with torch.no_grad():
        solve_and_add_scaled_vector_kernel[grid_size, block_size](
            A_ptr=A.data_ptr(),
            b_ptr=b.data_ptr(),
            y_ptr=y.data_ptr(),
            x_ptr=out.data_ptr(),
            n=n,
            k=k,
            alpha=alpha,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return out
