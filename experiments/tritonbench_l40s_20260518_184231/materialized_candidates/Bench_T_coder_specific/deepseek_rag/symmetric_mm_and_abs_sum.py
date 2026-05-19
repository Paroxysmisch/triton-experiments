import torch
import triton
import triton.language as tl

@triton.jit
def symmetric_mm_and_abs_sum_kernel(
        A_ptr,
        C_ptr,
        alpha_const: tl.constexpr,
        beta_const: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load data
    A = tl.load(A_ptr + offsets, mask=mask)
    C = tl.load(C_ptr + offsets, mask=mask)

    # Compute symmetric matrix multiplication and scaling
    product = tl.dot(A, A)
    result = alpha_const * product + beta_const * C

    # Store result
    tl.store(C_ptr + offsets, result, mask=mask)

    # Compute sum of absolute values
    asum = tl.sum(tl.abs(result))

    return asum

def symmetric_mm_and_abs_sum(
        A: torch.Tensor,
        C: torch.Tensor,
        alpha: float,
        beta: float,
) -> torch.Tensor:
    # Make tensors contiguous if needed
    if not A.is_contiguous():
        A = A.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    # Get shapes
    M, N = A.shape

    # Init output tensor
    asum: torch.Tensor = torch.tensor(0.0, device=A.device)

    # Call triton kernel
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    symmetric_mm_and_abs_sum_kernel[grid](A, C, alpha, beta, M, N, BLOCK_SIZE=1024)

    return asum
