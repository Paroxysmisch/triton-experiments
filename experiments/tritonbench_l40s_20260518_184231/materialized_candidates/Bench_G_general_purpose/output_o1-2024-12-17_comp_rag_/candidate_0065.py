import torch
import triton
import triton.language as tl

@triton.jit
def square_kernel(
    input_ptr,
    output_ptr,
    in_stride,
    out_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    row_input_ptr = input_ptr + row_id * in_stride
    row_output_ptr = output_ptr + row_id * out_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_data = tl.load(row_input_ptr + offsets, mask=mask, other=0.0)
    squared = row_data * row_data
    tl.store(row_output_ptr + offsets, squared, mask=mask)

def square(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, "Input must be a 2D tensor."
    n_rows, n_cols = x.shape

    # Determine BLOCK_SIZE as the smallest power of two >= n_cols
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE <<= 1

    # Heuristic for num_warps
    if BLOCK_SIZE >= 1024:
        num_warps = 4
    else:
        num_warps = 1

    y = torch.empty_like(x)

    grid = (n_rows,)
    square_kernel[grid](
        input_ptr=x,
        output_ptr=y,
        in_stride=x.stride(0),
        out_stride=y.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1
    )
    return y
