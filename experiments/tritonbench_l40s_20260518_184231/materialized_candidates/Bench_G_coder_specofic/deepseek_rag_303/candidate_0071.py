import triton
import triton.language as tl
import torch

@triton.jit
def square_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask)
    square_output = row * row
    tl.store(y_ptr + row_idx * y_row_stride + col_offsets, square_output, mask=mask)

def square(x: torch.Tensor):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = min(1024, 2 ** (tl.math.ilog2(n_cols) + 1))
    num_warps = 4
    y = torch.empty_like(x)
    square_kernel[(n_rows,)](x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y
