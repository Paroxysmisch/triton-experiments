import torch
import triton
import triton.language as tl

@triton.jit
def _symv_kernel(
    A_ptr, x_ptr, y_ptr,
    N, alpha, beta,
    strideA0, strideA1,
    strideX, strideY,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_idx < N
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over columns in steps of BLOCK_SIZE
    for offset in range(0, N, BLOCK_SIZE):
        col_idx = offset + tl.arange(0, BLOCK_SIZE)
        col_mask = col_idx < N
        # Load a block of A
        a_block = tl.load(
            A_ptr + row_idx[:, None] * strideA0 + col_idx[None, :] * strideA1,
            mask=mask[:, None] & col_mask[None, :],
            other=0
