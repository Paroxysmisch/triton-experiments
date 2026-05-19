import torch
import triton
import triton.language as tl

@triton.jit
def symmetric_mm_kernel(A_ptr, C_ptr, alpha, beta, n, m):
    # Define grid size
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Initialize accumulator
    acc = 0.0

    # Load A and compute A * A^T
    for k in range(m):
        a = tl.load(A_ptr + row * m + k)
        b = tl.load(A_ptr + col * m + k)
        acc += a * b

    # Scale the result by alpha
    acc *= alpha

    # Load C and scale it by beta
    c = tl.load(C_ptr + row * n + col) * beta

    # Update C with the new value
    result = acc + c
    tl.store(C_ptr + row * n + col, result)

@torch.no_grad()
def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, m = A.shape
    # Ensure C has the same shape as the result of alpha * torch.mm(A, A.T)
    assert C.shape == (n, n), "C must have shape (n, n)"

    # Launch the Triton kernel
    grid = (n, n)
    symmetric_mm_kernel[grid](A.data_ptr(), C.data_ptr(), alpha, beta, n, m)

    # Compute the sum of absolute values
    asum = torch.sum(torch.abs(C))
    return asum
