import triton
import triton.language as tl
import torch

@triton.jit
def square_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each kernel instance processes one row
    pid = tl.program_id(0)
    # Compute starting pointers for the current row
    row_start_input = input_ptr + pid * input_row_stride
    row_start_output = output_ptr + pid * output_row_stride
    # Generate column indices within BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out-of-bounds access
    mask = cols < n_cols
    # Load the row data, masking out-of-bounds columns
    x = tl.load(row_start_input + cols, mask=mask, other=0)
    # Compute square
    square = x * x
    # Store the result
    tl.store(row_start_output + cols, square, mask=mask)

def square(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("Input tensor must be 2D")
    n_rows, n_cols = x.shape
    # Determine the next power of two for BLOCK_SIZE
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE <<= 1
    # Adjust num_warps based on BLOCK_SIZE for optimal performance
    if BLOCK_SIZE >= 512:
        num_warps = 8
    else:
        num_warps = 4
    # Allocate output tensor
    y = torch.empty_like(x)
    # Grid configuration: one kernel instance per row
    grid = (n_rows,)
    # Launch kernel with appropriate parameters
    square_kernel[grid](
        x.data_ptr(),
        y.data_ptr(),
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y
