import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mv_kernel(
    A_ptr, x_ptr, output_ptr,
    n, stride_am, stride_an, stride_x,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row < n
    offs_col = tl.arange(0, BLOCK_SIZE)
    col_mask = offs_col < n

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for block_col in range(0, tl.cdiv(n, BLOCK_SIZE)):
        col = block_col * BLOCK_SIZE + offs_col
        a_ptrs = A_ptr + row[:, None] * stride_am + col[None, :] * stride_an
        x_ptrs = x_ptr + col * stride_x
        a = tl.load(a_ptrs, mask=mask[:, None] & col_mask[None, :], other=0.0)
        x = tl.load(x_ptrs, mask=col < n, other=0.0)
        accumulator += tl.sum(a * x, axis=1)

    output_ptrs = output_ptr + row * stride_x
    tl.store(output_ptrs, accumulator, mask=mask)

def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    assert A.is_contiguous() and x.is_contiguous(), "Inputs must be contiguous"
    n = A.size(0)
    assert A.size(1) == n and x.size(0) == n, "Dimension mismatch"
    
    mv_output = torch.empty_like(x)
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    _symmetric_mv_kernel[grid](
        A, x, mv_output,
        n,
        A.stride(0), A.stride(1),
        x.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    y = alpha * mv_output + beta * x
    return torch.norm(y, p)
