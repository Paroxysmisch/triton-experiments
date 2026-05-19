import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_matrix_vector_norm_kernel(
    A_ptr, x_ptr, y_ptr, alpha: tl.float32, beta: tl.float32, p: tl.float32, n: tl.int32, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load x and y
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Initialize result vector
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Matrix-vector product
    for i in range(n):
        a = tl.load(A_ptr + i * n + offsets, mask=mask)
        result += a * x[i]

    # Scale by alpha and add beta * y
    result = alpha * result + beta * y

    # Store the result back to y
    tl.store(y_ptr + offsets, result, mask=mask)

    # Compute the norm
    norm = tl.sum(tl.abs(result) ** p, axis=0) ** (1.0 / p)
    tl.atomic_add(y_ptr + 0, norm)

def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    """
    Computes the matrix-vector product for a symmetric matrix `A` and a vector `x`, with scaling factors `alpha` and `beta`.
    Then calculates the norm of the resulting vector `y`.

    :param A (torch.Tensor): A symmetric matrix of shape `(n, n)`.
    :param x (torch.Tensor): A vector of shape `(n,)`.
    :param alpha (float): Scalar multiplier for the matrix-vector product.
    :param beta (float): Scalar multiplier added to `y`.
    :param p (float, optional): Order of the norm. Default is 2.0 (Euclidean norm).
    :return (torch.Tensor): The norm of the resulting vector `y`.
    """
    n = A.size(0)
    y = torch.zeros(n, device=A.device, dtype=A.dtype)

    # Ensure inputs are contiguous
    if not A.is_contiguous():
        A = A.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    # Call the Triton kernel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _symmetric_matrix_vector_norm_kernel[grid](
        A, x, y, alpha, beta, p, n, BLOCK_SIZE=1024
    )

    # Return the norm of the resulting vector y
    return torch.norm(y, p)

# Example usage
A = torch.tensor([[1.0, 0.5], [0.5, 1.0]], device='cuda')
x = torch.tensor([1.0, 2.0], device='cuda')
alpha = 2.0
beta = 1.0
p = 2.0

norm = symmetric_matrix_vector_norm(A, x, alpha, beta, p)
print(norm)
