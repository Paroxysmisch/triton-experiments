import torch
import triton
import triton.language as tl

@triton.jit
def _tril_mm_kernel(A_ptr, B_ptr, C_ptr, alpha: float, beta: float, n: int, p: int, BLOCK_SIZE: tl.constexpr):
    """ Triton kernel for matrix multiplication of lower triangular part of A with B """
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask_row = row < n
    mask_col = col < p
    mask = mask_row[:, None] & mask_col[None, :]

    # Load lower triangular part of A
    A = tl.load(A_ptr + row[:, None] * n + col[None, :], mask=mask_row[:, None] & (col[None, :] <= row[:, None]))
    B = tl.load(B_ptr + col[None, :] * p + col[None, :], mask=mask_col)

    # Matrix multiplication
    C_temp = tl.dot(A, B) * alpha

    # Scale by beta
    C_final = C_temp * beta

    # Write-back result
    tl.store(C_ptr + row[:, None] * p + col[None, :], C_final, mask=mask)

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Wrapper function for matrix multiplication of lower triangular part of A with B
    :param A (torch.Tensor): A 2D matrix of shape (n, n)
    :param B (torch.Tensor): A matrix of shape (n, p)
    :param alpha (float): Scaling factor for the initial matrix multiplication result
    :param beta (float): Scaling factor for the final result
    :return (torch.Tensor): Result of the operation
    """
    n, p = B.shape
    # Init output tensor
    C = torch.empty((n, p), dtype=B.dtype, device=B.device)

    # Call Triton kernel
    grid = (triton.cdiv(n, 1024), triton.cdiv(p, 1024))  # Adjust grid size as needed
    _tril_mm_kernel[grid](A, B, C, alpha, beta, n, p, BLOCK_SIZE=1024)

    return C
