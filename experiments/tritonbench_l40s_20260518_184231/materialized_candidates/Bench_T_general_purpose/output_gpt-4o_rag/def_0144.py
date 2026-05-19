import torch
import triton
import triton.language as tl

@triton.jit
def _scaled_matmul_add_kernel(
        A_ptr, B_ptr, C_ptr, alpha, beta,
        n, m, p,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """ Triton kernel for scaled matrix multiplication and addition """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute the start of the block in the output matrix
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create a block of zeros for accumulation
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, m, BLOCK_SIZE_K):
        # Load blocks of A and B
        a = tl.load(A_ptr + block_start_m * m + k, mask=(block_start_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < n) & (k + tl.arange(0, BLOCK_SIZE_K) < m), other=0.0)
        b = tl.load(B_ptr + k * p + block_start_n, mask=(k + tl.arange(0, BLOCK_SIZE_K)[:, None] < m) & (block_start_n + tl.arange(0, BLOCK_SIZE_N) < p), other=0.0)

        # Accumulate the product
        acc += tl.dot(a, b)

    # Scale the result by alpha
    acc *= alpha

    # Load the corresponding block of C and scale by beta
    c = tl.load(C_ptr + block_start_m * p + block_start_n, mask=(block_start_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < n) & (block_start_n + tl.arange(0, BLOCK_SIZE_N) < p), other=0.0)
    c *= beta

    # Add the scaled result to C
    acc += c

    # Store the result back to C
    tl.store(C_ptr + block_start_m * p + block_start_n, acc, mask=(block_start_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < n) & (block_start_n + tl.arange(0, BLOCK_SIZE_N) < p))

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the Triton kernel that performs a scaled matrix-matrix product and calculates the dot product of the first two rows.
    :param A (torch.Tensor): First input matrix of shape `(n, m)`.
    :param B (torch.Tensor): Second input matrix of shape `(m, p)`.
    :param alpha (float): Scalar multiplier for the matrix-matrix product.
    :param beta (float): Scalar multiplier for the input matrix `C`.
    :param C (torch.Tensor): Output matrix of shape `(n, p)` where the results are added.
    :return (torch.Tensor): Dot product of the first two rows of the updated matrix C.
    """
    # Ensure matrices are contiguous
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    # Get dimensions
    n, m = A.shape
    _, p = B.shape

    # Define grid size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(n, BLOCK_SIZE_M), triton.cdiv(p, BLOCK_SIZE_N))

    # Call the Triton kernel
    _scaled_matmul_add_kernel[grid](
        A, B, C, alpha, beta,
        n, m, p,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    # Compute the dot product of the first two rows of C
    result = torch.dot(C[0], C[1])
    return result
