import torch
import triton
import triton.language as tl

@triton.jit
def square_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    input_row_ptr = input_ptr + pid * input_row_stride
    output_row_ptr = output_ptr + pid * output_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row = tl.load(input_row_ptr + offsets, mask=mask)
    square_output = row * row
    tl.store(output_row_ptr + offsets, square_output, mask=mask)

def square(x: torch.Tensor):
    assert x.dim() == 2, "Input must be 2D"
    n_rows, n_cols = x.shape
    # Calculate the smallest power of two greater than or equal to n_cols
    if n_cols == 0:
        BLOCK_SIZE = 1
    else:
        BLOCK_SIZE = 1 << (n_cols - 1).bit_length()
    # Adjust num_warps based on BLOCK_SIZE to optimize performance
    if BLOCK_SIZE <= 512:
        num_warps = 4
    else:
        num_warps = 8
    y = torch.empty_like(x)
    grid = (n_rows,)
    square_kernel[grid](
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y
