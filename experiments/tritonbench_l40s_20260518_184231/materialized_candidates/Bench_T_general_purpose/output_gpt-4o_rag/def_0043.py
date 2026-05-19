import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_matrix_vector_product_kernel(
    A_pointer,
    x_pointer,
    y_pointer,
    alpha: float,
    beta: float,
    n: int,
    BLOCK_SIZE: tl.constexpr
):
    """ Triton kernel for symmetric matrix-vector product """
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < n

    # Load the vector x
    x = tl.load(x_pointer + row_offsets, mask=mask)

    # Initialize y with zeros
    y = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Perform the matrix-vector product
    for i in range(0, n):
        # Load the symmetric matrix row
        A_row = tl.load(A_pointer + row_offsets * n + i, mask=mask)
        # Update y
        y += A_row * x[i]

    # Scale and add beta * y
    y = alpha * y + beta * tl.load(y_pointer + row_offsets, mask=mask)

    # Store the result back to y
    tl.store(y_pointer + row_offsets, y, mask=mask)

def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    """
    Wrapper function for symmetric matrix-vector product and norm computation
    :param A (torch.Tensor): A symmetric matrix of shape (n, n)
    :param x (torch.Tensor): A vector of shape (n,)
    :param alpha (float): Scalar multiplier for the matrix-vector product
    :param beta (float): Scalar multiplier added to y
    :param p (float, optional): Order of the norm. Default is 2.0 (Euclidean norm)
    :return (torch.Tensor): Norm of the resulting vector y
    """
    n = A.shape[0]
    assert A.shape == (n, n), "A must be a square matrix"
    assert x.shape == (n,), "x must be a vector of shape (n,)"

    # Initialize y as a zero vector
    y = torch.zeros_like(x)

    # Make sure A and x are contiguous
    if not A.is_contiguous():
        A = A.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()

    # Call Triton kernel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _symmetric_matrix_vector_product_kernel[grid](A, x, y, alpha, beta, n, BLOCK_SIZE=1024)

    # Compute the norm of y
    norm = torch.norm(y, p)

    return norm
