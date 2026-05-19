import torch
import triton
import triton.language as tl

@triton.jit
def symmetric_mm_kernel(A_ptr, C_ptr, alpha, beta, n, m, stride_am, stride_an, stride_cm, stride_cn, BLOCK_SIZE: tl.constexpr):
    # Get the block indices
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    # Initialize accumulators
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Load matrix A in tiles
    for k in range(0, m, BLOCK_SIZE):
        a_tile = tl.load(A_ptr + row_idx * BLOCK_SIZE * stride_am + k * stride_an + tl.arange(0, BLOCK_SIZE)[:, None] * stride_an + tl.arange(0, BLOCK_SIZE)[None, :] * stride_am, mask=(tl.arange(0, BLOCK_SIZE)[:, None] < n) & (tl.arange(0, BLOCK_SIZE)[None, :] < m), other=0.0)
        a_tile_T = tl.load(A_ptr + col_idx * BLOCK_SIZE * stride_an + k * stride_am + tl.arange(0, BLOCK_SIZE)[:, None] * stride_am + tl.arange(0, BLOCK_SIZE)[None, :] * stride_an, mask=(tl.arange(0, BLOCK_SIZE)[:, None] < n) & (tl.arange(0, BLOCK_SIZE)[None, :] < m), other=0.0)
        
        # Accumulate the product
        acc += tl.dot(a_tile, a_tile_T)

    # Scale by alpha and add beta-scaled C
    c_tile = tl.load(C_ptr + row_idx * BLOCK_SIZE * stride_cm + col_idx * BLOCK_SIZE * stride_cn + tl.arange(0, BLOCK_SIZE)[:, None] * stride_cn + tl.arange(0, BLOCK_SIZE)[None, :] * stride_cm, mask=(tl.arange(0, BLOCK_SIZE)[:, None] < n) & (tl.arange(0, BLOCK_SIZE)[None, :] < n), other=0.0)
    result = alpha * acc + beta * c_tile

    # Store the result back to C
    tl.store(C_ptr + row_idx * BLOCK_SIZE * stride_cm + col_idx * BLOCK_SIZE * stride_cn + tl.arange(0, BLOCK_SIZE)[:, None] * stride_cn + tl.arange(0, BLOCK_SIZE)[None, :] * stride_cm, result, mask=(tl.arange(0, BLOCK_SIZE)[:, None] < n) & (tl.arange(0, BLOCK_SIZE)[None, :] < n))

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.dim() == 2, "A must be a 2D matrix"
    assert C.dim() == 2, "C must be a 2D matrix"
    n, m = A.shape
    assert C.shape == (n, n), "C must have the shape (n, n)"

    # Define block size for Triton kernel
    BLOCK_SIZE = 16

    # Launch Triton kernel
    grid = (n // BLOCK_SIZE, n // BLOCK_SIZE)
    symmetric_mm_kernel[grid](
        A, C, alpha, beta, n, m,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Calculate the sum of absolute values
    asum = torch.sum(torch.abs(C))
    return asum
