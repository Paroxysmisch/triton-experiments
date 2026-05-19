import triton
import triton.language as tl
import torch

@triton.jit
def square_kernel(
    x_ptr,  # Pointer to the input tensor
    y_ptr,  # Pointer to the output tensor
    row_stride_x,  # Row stride of the input tensor
    row_stride_y,  # Row stride of the output tensor
    n_cols,  # Number of columns in the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size for memory management
):
    # Compute the starting index for the row
    row_id = tl.program_id(0)
    x_row_ptr = x_ptr + row_id * row_stride_x
    y_row_ptr = y_ptr + row_id * row_stride_y

    # Load the row into SRAM
    x_row = tl.load(x_row_ptr, mask=row_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < n_cols, other=0.0)

    # Compute the square of each element
    square_output = x_row * x_row

    # Store the squared values back to global memory
    tl.store(y_row_ptr, square_output, mask=row_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < n_cols)

def square(x: torch.Tensor) -> torch.Tensor:
    # Get the number of rows and columns
    n_rows, n_cols = x.shape

    # Determine the block size as the smallest power of two greater than the number of columns
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE *= 2

    # Set the number of warps based on the block size
    num_warps = 4 if BLOCK_SIZE <= 256 else 8

    # Allocate the output tensor
    y = torch.empty_like(x)

    # Launch the kernel
    square_kernel[(n_rows,)](x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE, num_warps=num_warps)

    return y
