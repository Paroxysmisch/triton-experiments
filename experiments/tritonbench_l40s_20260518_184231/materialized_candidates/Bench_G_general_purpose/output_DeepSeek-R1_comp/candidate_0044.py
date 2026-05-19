import torch
import triton
import triton.language as tl

def next_power_of_two(n):
    """Returns the smallest power of two greater than or equal to n"""
    if n == 0:
        return 1
    return 1 << (n - 1).bit_length()

@triton.jit
def softmax_kernel(
    x_ptr,  # Pointer to input tensor
    y_ptr,  # Pointer to output tensor
    stride_x_row,  # Stride between rows for input
    stride_x_col,  # Stride between columns for input
    stride_y_row,  # Stride between rows for output
    stride_y_col,  # Stride between columns for output
    N_COLS,  # Number of columns in the input tensor
    BLOCK_SIZE: tl.constexpr,  # Block size (power of two)
):
    # Get row index for 1D grid
    row_idx = tl.program_id(0)
    
    # Compute pointers to the current row
    row_start_x = x_ptr + row_idx * stride_x_row
    row_start_y = y_ptr + row_idx * stride_y_row
    
    # Generate column offsets and mask for valid columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS
    
    # Load row data into SRAM with masking
    x_ptrs = row_start_x + col_offsets * stride_x_col
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    
    # Numerical stability: subtract maximum for exponentiation
    x_max = tl.max(x, axis=0)
    x_sub = x - x_max
    
    # Compute softmax components
    numerator = tl.exp(x_sub)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator
    
    # Store results with same masking
    y_ptrs = row_start_y + col_offsets * stride_y_col
    tl.store(y_ptrs, softmax_out, mask=mask)

def softmax(x: torch.Tensor):
    # Ensure 2D input and get dimensions
    assert x.dim() == 2, "Input tensor must be 2D"
    M, N = x.shape
    
    # Compute optimal block size and warp configuration
    BLOCK_SIZE = next_power_of_two(N)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8  # Use more warps for larger blocks
    
    # Create output tensor with same shape as input
    y = torch.empty_like(x)
    
    # Launch 1D grid where each row is handled by one kernel instance
    grid = (M,)
    softmax_kernel[grid](
        x, y,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        N_COLS=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return y
