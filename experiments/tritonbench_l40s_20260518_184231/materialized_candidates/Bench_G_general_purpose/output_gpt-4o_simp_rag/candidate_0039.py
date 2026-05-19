import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # Get the program's row index
    row_idx = tl.program_id(0)
    # Calculate the starting pointer for the current row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # Create offsets for columns within the block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Compute pointers for loading the row data
    input_ptrs = row_start_ptr + col_offsets
    # Create a mask for valid columns
    mask = col_offsets < n_cols
    # Load the row with masking for valid columns
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    # Subtract the maximum value in the row for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Compute exponentials
    numerator = tl.exp(row_minus_max)
    # Sum the exponentials to get the denominator
    denominator = tl.sum(numerator, axis=0)
    # Compute softmax output
    softmax_output = numerator / denominator
    # Calculate the starting pointer for the output row
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    # Compute pointers for storing the result
    output_ptrs = output_row_start_ptr + col_offsets
    # Store the softmax output
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x):
    # Extract the number of rows and columns from the input tensor
    n_rows, n_cols = x.shape
    # Determine the block size, which is the next power of 2 greater than the number of columns
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Determine the number of warps to use
    num_warps = 8
    # Create an empty tensor to store the output
    y = torch.empty_like(x)
    # Launch the kernel
    grid = (n_rows,)
    softmax_kernel[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return y

# Example usage
torch.manual_seed(42)
x = torch.randn(1823, 781, device='cuda')
y_triton = softmax(x)
