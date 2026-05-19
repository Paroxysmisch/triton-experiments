import torch
import triton
import triton.language as tl

@triton.jit
def tril_mm_and_scale_kernel(
        A_ptr, B_ptr, C_ptr,
        alpha: float, beta: float,
        n: int, p: int,
        BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for tril matrix multiplication and scaling"""
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Load the row from A and the column from B
    A_row = tl.load(A_ptr + row_start * n + col_offsets, mask=col_offsets < n, other=0.0)
    B_col = tl.load(B_ptr + col_offsets[:, None] * p, mask=col_offsets[:, None] < p, other=0.0)

    # Compute the product and apply alpha
    result = tl.dot(A_row, B_col) * alpha

    # Scale the result by beta and store it
    result *= beta
    tl.store(C_ptr + row_start * p + col_offsets, result, mask=col_offsets < p)

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Wrapper function for tril_mm_and_scale triton kernel
    :param A (torch.Tensor): A 2D matrix of shape (n, n)
    :param B (torch.Tensor): A 2D matrix of shape (n, p)
    :param alpha (float): Scaling factor for the initial matrix multiplication result
    :param beta (float): Scaling factor for the final result
    :return (torch.Tensor): The result of the scaled matrix multiplication
    """
    n, p = A.shape[0], B.shape[1]
    C = torch.empty((n, p), dtype=A.dtype, device=A.device)

    # Ensure A is lower triangular
    A = torch.tril(A)

    # Ensure inputs are contiguous
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    tril_mm_and_scale_kernel[grid](A, B, C, alpha, beta, n, p, BLOCK_SIZE=128)

    return C
