import torch
import triton
import triton.language as tl

@triton.jit
def tril_mm_and_scale_kernel(
    A_ptr, B_ptr, C_ptr,
    alpha, beta,
    n, p,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the row and column indices for the current thread
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a pointer for the block of A and B
    A_block_ptr = A_ptr + row[:, None] * n + col[None, :]
    B_block_ptr = B_ptr + col[:, None] * p + tl.arange(0, BLOCK_SIZE)[None, :]
    
    # Load blocks of A and B
    A_block = tl.load(A_block_ptr, mask=(row[:, None] < n) & (col[None, :] < n))
    B_block = tl.load(B_block_ptr, mask=(col[:, None] < n) & (tl.arange(0, BLOCK_SIZE)[None, :] < p))
    
    # Compute the lower triangular mask
    tril_mask = row[:, None] >= col[None, :]
    
    # Zero out the upper triangular part of A
    A_block = tl.where(tril_mask, A_block, 0.0)
    
    # Perform matrix multiplication
    C_block = tl.dot(A_block, B_block)
    
    # Scale the result by alpha
    C_block *= alpha
    
    # Scale the final result by beta
    C_block *= beta
    
    # Store the result back to C
    C_ptr = C_ptr + row[:, None] * p + tl.arange(0, BLOCK_SIZE)[None, :]
    tl.store(C_ptr, C_block, mask=(row[:, None] < n) & (tl.arange(0, BLOCK_SIZE)[None, :] < p))

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D matrices"
    n, n_ = A.shape
    n_b, p = B.shape
    assert n == n_, "Matrix A must be square"
    assert n == n_b, "The number of rows in B must match the number of rows in A"
    
    # Allocate output matrix C
    C = torch.empty((n, p), device=A.device, dtype=A.dtype)
    
    # Launch the Triton kernel
    BLOCK_SIZE = 32  # You can tune this parameter
    grid = (triton.cdiv(n, BLOCK_SIZE), triton.cdiv(p, BLOCK_SIZE))
    
    tril_mm_and_scale_kernel[grid](
        A, B, C,
        alpha, beta,
        n, p,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return C
