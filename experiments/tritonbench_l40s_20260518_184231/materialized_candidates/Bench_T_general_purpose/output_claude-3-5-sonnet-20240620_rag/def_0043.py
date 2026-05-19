import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_vector_product_kernel(A_ptr, x_ptr, y_ptr, alpha: float, beta: float, n: int, BLOCK_SIZE: tl.constexpr):
    """ Triton kernel for matrix-vector product """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load matrix A and vector x
    A = tl.load(A_ptr + (offsets[:, None] * n + tl.arange(0, n))[None, :].flatten(), mask=mask)
    x = tl.load(x_ptr, mask=mask)

    # Compute y = alpha * A @ x + beta * y
    y = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    for i in range(n):
        y += A[i] * x[i]
    y = alpha * y + beta * tl.load(y_ptr + offsets, mask=mask)

    # Write-back output
    tl.store(y_ptr + offsets, y, mask=mask)

def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    """
    Computes the matrix-vector product for a symmetric matrix A and a vector x,
    with scaling factors alpha and beta, and then calculates the norm of the resulting vector y.
    
    :param A (torch.Tensor): A symmetric matrix of shape (n, n).
    :param x (torch.Tensor): A vector of shape (n,).
    :param alpha (float): Scalar multiplier for the matrix-vector product.
    :param beta (float): Scalar multiplier added to y.
    :param p (float, optional): Order of the norm. Default is 2.0 (Euclidean norm).
    :return (torch.Tensor): The norm of the resulting vector y.
    """
    n = A.shape[0]
    y = torch.zeros(n, dtype=A.dtype, device=A.device)

    # Call Triton kernel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _matrix_vector_product_kernel[grid](A, x, y, alpha, beta, n, BLOCK_SIZE=1024)

    # Calculate the norm
    norm = torch.norm(y, p)
    return norm
