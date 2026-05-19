import torch
import triton
import triton.language as tl

@triton.jit
def load_reduce_kernel(
    x_ptr, 
    y_ptr,
    stride_xm,
    stride_xn,
    stride_y,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.full((BLOCK_M,), -float('inf'))
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = (row_offsets < M) & (col_offsets < N)
        x = tl.load(
            x_ptr + row_offsets[:, None] * stride_xm + col_offsets[None, :] * stride_xn,
            mask=mask,
            other=-float('inf')
        )
        row_max = tl.max(x, 1)
        acc = tl.maximum(acc, row_max)
    tl.store(
        y_ptr + row_offsets * stride_y,
        acc,
        mask=row_offsets < M
    )

def load_reduce():
    BLOCK_M = 128
    BLOCK_N = 128
    M = 256
    N = 512
    x = torch.randn((M, N), device='cuda')
    y = torch.empty((M,), device='cuda')
    grid = ((M + BLOCK_M - 1) // BLOCK_M,)
    load_reduce_kernel[grid](
        x, y,
        x.stride(0), x.stride(1),
        y.stride(0),
        M, N,
        BLOCK_M, BLOCK_N
    )
    torch.testing.assert_close(y, x.max(dim=1).values)
