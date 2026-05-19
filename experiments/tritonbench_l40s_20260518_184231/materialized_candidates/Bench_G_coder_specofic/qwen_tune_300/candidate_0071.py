import torch
import triton
import triton.language as tl
import math

@triton.jit
def square_kernel(
    in_ptr, out_ptr, in_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr
):
    # Launch one instance per row of the input
    row_idx = tl.program_id(0)
    in_row_start_ptr = in_ptr + row_idx * in_row_stride
    out_row_start_ptr = out_ptr + row_idx * out_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    in_ptrs = in_row_start_ptr + col_offsets
    out_ptrs = out_row_start_ptr + col_offsets

    # Load the row into SRAM, automatically handling end-of-row padding
    row_mask = col_offsets < n_cols
    row = tl.load(in_ptrs, mask=row_mask, other=0)
    square_output = row * row
    tl.store(out_ptrs, square_output, mask=row_mask)

def square(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, "Input tensor must be 2D"
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 2 ** (math.ceil(math.log2(n_cols)))
    num_warps = 4
    if BLOCK_SIZE >= 8192:
        num_warps = 32
    elif BLOCK_SIZE >= 4096:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    y = torch.empty_like(x)
    grid = lambda meta: (n_rows,)
    square_kernel[grid](x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y
