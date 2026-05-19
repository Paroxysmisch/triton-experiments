import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mm_kernel(
        A_ptr, C_ptr, alpha, beta, n, m, BLOCK_SIZE: tl.constexpr
):
    """ Triton kernel for symmetric matrix multiplication and scaling """
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE

    # Offsets for the block
    offsets = tl.arange(0, BLOCK_SIZE)
    row_offsets = row_start + offsets

    # Ensure we don't go out of bounds
    mask = row_offsets < n

    # Load a block of A
    A_block = tl.load(A_ptr + row_offsets[:, None] * m + offsets[None, :], mask=mask[:, None])

    # Compute A * A.T for this block
    result = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k in range(m):
        a_k = A_block[:, k]
        result += a_k[:, None] * a_k[None, :]

    # Scale by alpha
    result *= alpha

    # Load corresponding block of C and scale by beta
    C_block = tl.load(C_ptr + row_offsets[:, None] * n + offsets[None, :], mask=mask[:, None])
    C_block *= beta

    # Add results
    C_block += result

    # Write back the result
    tl.store(C_ptr + row_offsets[:, None] * n + offsets[None, :], C_block, mask=mask[:, None])

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Wrapper function for symmetric matrix multiplication and absolute sum
    :param A (torch.Tensor): Input matrix of shape (n, m)
    :param C (torch.Tensor): Matrix of shape (n, n)
    :param alpha (float): Scaling factor for the matrix product
    :param beta (float): Scaling factor for matrix C
    :return (torch.Tensor): Scalar tensor representing the sum of absolute values of the resulting matrix C
    """
    n, m = A.shape
    assert C.shape == (n, n), "Matrix C must have shape (n, n)"

    # Make sure matrices are contiguous
    if not A.is_contiguous():
        A = A.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    # Call the Triton kernel
    BLOCK_SIZE = 128  # You can choose an appropriate block size
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _symmetric_mm_kernel[grid](A, C, alpha, beta, n, m, BLOCK_SIZE=BLOCK_SIZE)

    # Compute and return the sum of absolute values
    return torch.sum(torch.abs(C))

# Example usage
A = torch.randn(256, 128, dtype=torch.float32, device='cuda')
C = torch.randn(256, 256, dtype=torch.float32, device='cuda')
alpha = 1.0
beta = 0.5

result = symmetric_mm_and_abs_sum(A, C, alpha, beta)
print(result)
