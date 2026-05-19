import triton
import triton.language as tl

@triton.jit
def mv_kernel(A_ptr, x_ptr, y_ptr, alpha, beta, n, m):
    """
    Computes the matrix-vector product y = alpha * torch.mv(A, x) + beta * y.

    Args:
    A_ptr: Pointer to the input matrix A of shape (n, m).
    x_ptr: Pointer to the input vector x of shape (m,).
    y_ptr: Pointer to the target vector y of shape (n,).
    alpha: Scalar multiplier for torch.mv(A, x).
    beta: Scalar multiplier for y.
    n: Number of rows in matrix A.
    m: Number of columns in matrix A.
    """
    pid = tl.program_id(axis=0)
    row_start = pid * n
    row_end = min(row_start + n, n)
    
    for i in range(row_start, row_end):
        acc = 0.0
        for j in range(m):
            acc += A_ptr[i * m + j] * x_ptr[j]
        y_ptr[i] = alpha * acc + beta * y_ptr[i]
