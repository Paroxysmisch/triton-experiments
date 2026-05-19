import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_multiply_kernel(A_ptr, B_ptr, C_ptr, alpha: float, beta: float, n_rows: int, n_cols: int, n_inner: int, BLOCK_SIZE: tl.constexpr):
    """ Triton kernel for matrix multiplication and addition """
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Initialize accumulator
    acc = 0.0
    
    # Compute the matrix product
    for k in range(n_inner):
        acc += tl.load(A_ptr + row * n_inner + k) * tl.load(B_ptr + k * n_cols + col)
    
    # Scale and add to C
    if row < n_rows and col < n_cols:
        C_val = tl.load(C_ptr + row * n_cols + col)
        tl.store(C_ptr + row * n_cols + col, alpha * acc + beta * C_val)

@triton.jit
def _row_dot_kernel(C_ptr, result_ptr, n_cols: int):
    """ Triton kernel for computing the dot product of the first two rows of C """
    row = tl.program_id(0)
    
    if row < 2:  # Only compute for the first two rows
        acc = 0.0
        for col in range(n_cols):
            acc += tl.load(C_ptr + row * n_cols + col) * tl.load(C_ptr + (row + 1) * n_cols + col)
        tl.store(result_ptr + row, acc)

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    """
    Computes a scaled matrix-matrix product and the dot product of the first two rows of the resulting matrix.
    
    :param A (torch.Tensor): First input matrix of shape `(n, m)`.
    :param B (torch.Tensor): Second input matrix of shape `(m, p)`.
    :param alpha (float): Scalar multiplier for the matrix-matrix product.
    :param beta (float): Scalar multiplier for the input matrix `C`.
    :param C (torch.Tensor): Output matrix of shape `(n, p)` where the results are added.
    :return (torch.Tensor): Dot product of the first two rows of the updated matrix `C`.
    """
    n_rows, n_inner = A.shape
    n_cols = B.shape[1]
    
    # Ensure C is contiguous
    if not C.is_contiguous():
        C = C.contiguous()
    
    # Call the matrix multiplication kernel
    grid = (triton.cdiv(n_rows, 1), triton.cdiv(n_cols, 1))
    _matrix_multiply_kernel[grid](A, B, C, alpha, beta, n_rows, n_cols, n_inner, BLOCK_SIZE=1024)
    
    # Prepare to compute the dot product
    result = torch.empty(1, dtype=C.dtype, device=C.device)
    _row_dot_kernel[(1,)](C, result, n_cols)
    
    return result
