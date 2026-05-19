import triton
import triton.language as tl
import torch

@triton.jit
def square_kernel(
    in_ptr, out_ptr,
    stride_in, stride_out,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    in_row_ptr = in_ptr + row_id * stride_in
    out_row_ptr = out_ptr + row_id * stride_out
    mask = col_offsets < n_cols

    row_vals = tl.load(in_row_ptr + col_offsets, mask=mask, other=0.0)
    square_output = row_vals * row_vals
    tl.store(out_row_ptr + col_offsets, square_output, mask=mask)

def square(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape

    # Determine the smallest power of two >= n_cols
    BLOCK_SIZE = 1 << (n_cols - 1).bit_length()

    # Adjust num_warps based on BLOCK_SIZE
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 4
    else:
        num_warps = 1

    y = torch.empty_like(x)

    grid = lambda meta: (n_rows,)
    square_kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE,
        num_warps=num_warps
    )
    return y
