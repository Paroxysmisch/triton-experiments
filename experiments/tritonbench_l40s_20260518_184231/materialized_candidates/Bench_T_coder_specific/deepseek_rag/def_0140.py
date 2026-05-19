import torch
import triton
import triton.language as tl

@triton.jit
def tril_mm_and_scale_kernel(
        A_ptr,
        B_ptr,
        alpha_beta_ptr,
        C_ptr,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    A = tl.load(A_ptr + offsets, mask=mask)
    B = tl.load(B_ptr + offsets, mask=mask)
    alpha, beta = tl.load(alpha_beta_ptr)
    # Perform matrix multiplication
    temp = tl.dot(A, B)
    C = alpha * temp
    C = beta * C
    # Write-back output
    tl.store(C_ptr + offsets, C, mask=mask)

def tril_mm_and_scale_triton(
        A: torch.Tensor,
        B: torch.Tensor,
        alpha: float,
        beta: float
) -> torch.Tensor:
    # Init output tensor
    C: torch.Tensor = torch.empty_like(B)
    alpha_beta = torch.tensor([alpha, beta])
    # Make input contiguous if needed
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    # Get number of elements in input
    number_of_elements: int = B.numel()
    # Call triton kernel
    grid = lambda meta: (triton.cdiv(number_of_elements, meta['BLOCK_SIZE']),)
    tril_mm_and_scale_kernel[grid](A, B, alpha_beta, C, number_of_elements, BLOCK_SIZE=1024)
    return C
