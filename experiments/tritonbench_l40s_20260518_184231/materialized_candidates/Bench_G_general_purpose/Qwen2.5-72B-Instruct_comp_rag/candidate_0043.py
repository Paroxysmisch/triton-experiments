import torch
import triton
import triton.language as tl

# Define the softmax kernel
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    # Get the current row index
    row_idx = tl.program_id(0)
    
    # Calculate the starting pointer for the current row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    
    # Load the row data into SRAM
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Subtract the maximum value for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    
    # Compute the exponentials (numerator)
    numerator = tl.exp(row_minus_max)
    
    # Compute the denominator (sum of exponentials)
    denominator = tl.sum(numerator, axis=0)
    
    # Compute the softmax output
    softmax_output = numerator / denominator
    
    # Store the output back to the output tensor
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

# Define the softmax function
def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if n_cols > 200_000 else 2
    
    # Create an empty output tensor
    y = torch.empty_like(x)
    
    # Configure the grid and block sizes
    grid = (n_rows, 1, 1)
    
    # Launch the kernel
    softmax_kernel[grid](
        y,  # Output tensor
        x,  # Input tensor
        x.stride(0),  # Input row stride
        y.stride(0),  # Output row stride
        n_rows,  # Number of rows
        n_cols,  # Number of columns
        BLOCK_SIZE=BLOCK_SIZE,  # Block size
        num_warps=num_warps,  # Number of warps
        num_stages=num_stages  # Number of stages
    )
    
    return y

# Test the softmax function
torch.manual_seed(42)
x = torch.randn(1823, 781, device='cuda')
y_triton = softmax(x)

# Print the first few elements of the result for verification
print(y_triton[:5, :5])
