import triton
import triton.language as tl

@triton.jit
def sum_dim_kernel(X, Sum, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Sum = Sum + pid
    row_mask = pid < M

    # Initialize sum for this block
    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += a
    tl.store(Sum, _sum, row_mask)
