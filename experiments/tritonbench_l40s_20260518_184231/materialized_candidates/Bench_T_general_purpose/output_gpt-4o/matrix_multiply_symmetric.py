import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiply_symmetric_kernel(
    A_ptr, B_ptr, C_ptr, alpha, beta, 
    n, m, p, stride_am, stride_an, stride_bm, stride_bp, stride_cp, stride_cn,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Create block IDs for row and column
    row_block_id = pid // (p // BLOCK_SIZE)
    col_block_id = pid % (p // BLOCK_SIZE)

    # Define pointers to the block
    A_block_ptr = A_ptr + row_block_id * BLOCK_SIZE * stride_an
    B_block_ptr = B_ptr + col_block_id * BLOCK_SIZE * stride_bp
    C_block_ptr = C_ptr + row_block_id * BLOCK_SIZE * stride_cn + col_block_id * BLOCK_SIZE * stride_cp

    # Load blocks of A, B, C
    A = tl.load(A_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_am + tl.arange(0, BLOCK_SIZE)[None, :] * stride_an, mask=(tl.arange(0, BLOCK_SIZE)[:, None] < n) & (tl.arange(0, BLOCK_SIZE)[None, :] < m), other=0)
    B = tl.load(B_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_bm + tl.arange(0, BLOCK_SIZE)[None, :] * stride_bp, mask=(tl.arange(0, BLOCK_SIZE)[:, None] < m) & (tl.arange(0, BLOCK_SIZE)[None, :] < p), other=0)
    C = tl.load(C_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_cp + tl.arange(0, BLOCK_SIZE)[None, :] * stride_cn, mask=(tl.arange(0, BLOCK_SIZE)[:, None] < n) & (tl.arange(0, BLOCK_SIZE)[None, :] < p), other=0)

    # Perform the first operation: C = alpha * A @ B + beta * C
    C_new = alpha * tl.dot(A, B) + beta * C

    # Store the result back
    tl.store(C_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_cp + tl.arange(0, BLOCK_SIZE)[None, :] * stride_cn, C_new, mask=(tl.arange(0, BLOCK_SIZE)[:, None] < n) & (tl.arange(0, BLOCK_SIZE)[None, :] < p))

    # Reload updated C for the next operation
    C = tl.load(C_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_cp + tl.arange(0, BLOCK_SIZE)[None, :] * stride_cn, mask=(tl.arange(0, BLOCK_SIZE)[:, None] < n) & (tl.arange(0, BLOCK_SIZE)[None, :] < p), other=0)

    # Perform the second operation: C = alpha * C @ C.T + beta * C
    C_transpose = tl.transpose(C)
    C_final = alpha * tl.dot(C, C_transpose) + beta * C

    # Store the final result back
    tl.store(C_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_cp + tl.arange(0, BLOCK_SIZE)[None, :] * stride_cn, C_final, mask=(tl.arange(0, BLOCK_SIZE)[:, None] < n) & (tl.arange(0, BLOCK_SIZE)[None, :] < p))

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Check input dimensions
    assert A.shape[1] == B.shape[0], "Inner dimensions of A and B must match."
    assert A.shape[0] == C.shape[0] and B.shape[1] == C.shape[1], "C must be of shape (n, p)."

    n, m = A.shape
    _, p = B.shape

    # Define block size
    BLOCK_SIZE = 16  # Example block size, tune this based on your hardware

    # Launch the Triton kernel
    grid = (n // BLOCK_SIZE) * (p // BLOCK_SIZE)
    matrix_multiply_symmetric_kernel[grid](
        A, B, C, alpha, beta, n, m, p,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )

    return C
