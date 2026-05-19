import triton
import triton.language as tl
import torch

# Kernel for matrix-matrix multiplication and symmetric update
@triton.jit
def matmul_update_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, n, m, p, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    A = tl.load(A_ptr + offs_m[:, None] * m + tl.arange(0, m), mask=offs_m[:, None] < n)
    B = tl.load(B_ptr + tl.arange(0, m)[:, None] * p + offs_n, mask=offs_n < p)

    C = tl.dot(A, B)
    C = alpha * C + beta * tl.load(C_ptr + offs_m[:, None] * p + offs_n, mask=(offs_m[:, None] < n) & (offs_n < p))
    tl.store(C_ptr + offs_m[:, None] * p + offs_n, C, mask=(offs_m[:, None] < n) & (offs_n < p))

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, m = A.shape
    _, p = B.shape
    assert C.shape == (n, p), "C must have shape (n, p)"

    # Allocate result tensor
    result = C.clone()

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    # Grid size for the kernel
    grid = (triton.cdiv(n, BLOCK_SIZE_M), triton.cdiv(p, BLOCK_SIZE_N))

    # First operation: C = alpha * torch.mm(A, B) + beta * C
    matmul_update_kernel[grid](A, B, result, alpha, beta, n, m, p, BLOCK_SIZE_M, BLOCK_SIZE_N)

    # Second operation: C = alpha * torch.mm(C, C.T) + beta * C
    grid = (triton.cdiv(n, BLOCK_SIZE_M), triton.cdiv(n, BLOCK_SIZE_N))
    matmul_update_kernel[grid](result, result.T, result, alpha, beta, n, p, n, BLOCK_SIZE_M, BLOCK_SIZE_N)

    return result

# Example usage
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
B = torch.tensor([[0.5, -1.0], [1.5, 2.0]], dtype=torch.float32)
C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
alpha, beta = 2.0, 0.5
result = matrix_multiply_symmetric(A, B, C, alpha, beta)
print(result)
