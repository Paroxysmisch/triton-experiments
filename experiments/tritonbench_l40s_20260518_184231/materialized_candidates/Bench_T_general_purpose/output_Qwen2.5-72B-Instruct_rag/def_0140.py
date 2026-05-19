import torch
import triton
import triton.language as tl

@triton.jit
def tril_mm_and_scale_kernel(
    A_ptr,  # Pointer to matrix A
    B_ptr,  # Pointer to matrix B
    C_ptr,  # Pointer to output matrix C
    alpha,  # Scaling factor for the initial matrix multiplication result
    beta,   # Scaling factor for the final result
    n,      # Dimension of matrix A (n x n)
    p,      # Dimension of matrix B (n x p)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(n, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(p, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_n)

    # Offsets for A, B, and C
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    A = A_ptr + (offs_am[:, None] * n + offs_k[None, :])
    B = B_ptr + (offs_k[:, None] * p + offs_bn[None, :])
    C = C_ptr + (offs_am[:, None] * p + offs_bn[None, :])

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Compute the matrix multiplication
    for k in range(0, n, BLOCK_SIZE_K):
        # Load the blocks of A and B
        a = tl.load(A + k * n, mask=offs_am[:, None] >= offs_k[None, :], other=0.0)
        b = tl.load(B + k * p, mask=offs_k[:, None] < p, other=0.0)
        # Perform the matrix multiplication
        acc += tl.dot(a, b)

    # Scale the result by alpha
    acc *= alpha

    # Store the result in C
    tl.store(C, acc.to(C_ptr.dtype.element_ty), mask=offs_am[:, None] < p)

    # Scale the final result by beta
    acc *= beta
    tl.store(C, acc.to(C_ptr.dtype.element_ty), mask=offs_am[:, None] < p)

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Wrapper function for the Triton kernel that performs matrix multiplication of the lower triangular part of A with B,
    scales the result by alpha, and then scales the final output by beta.

    :param A (torch.Tensor): A 2D matrix to be multiplied, of shape (n, n).
    :param B (torch.Tensor): A matrix to be multiplied with the lower triangular part of A, of shape (n, p).
    :param alpha (float): Scaling factor for the initial matrix multiplication result.
    :param beta (float): Scaling factor for the final result.
    :return (torch.Tensor): The final result of the operation.
    """
    # Ensure A and B are contiguous
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    # Get dimensions
    n, p = A.shape[0], B.shape[1]

    # Allocate output tensor
    C = torch.empty((n, p), device=A.device, dtype=A.dtype)

    # Define block and grid sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 8

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE_M']) * triton.cdiv(p, meta['BLOCK_SIZE_N']),)
    tril_mm_and_scale_kernel[grid](
        A, B, C, alpha, beta, n, p,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )

    return C

import torch

# Test data
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device='cuda')
B = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32, device='cuda')
alpha = 2.0
beta = 0.5

# Call the wrapper function
C = tril_mm_and_scale(A, B, alpha, beta)

# Expected result
expected_C = 0.5 * (2.0 * torch.mm(torch.tril(A), B))

# Check if the results match
print("C:", C)
print("Expected C:", expected_C)
print("Results match:", torch.allclose(C, expected_C))
