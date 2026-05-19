import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    # Get the row index for the current program
    row_idx = tl.program_id(0)
    
    # Compute the starting pointer for the current row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Create a range for the columns in the row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Compute the input pointers for the current row
    input_ptrs = row_start_ptr + col_offsets
    
    # Create a mask to handle cases where the number of columns is less than BLOCK_SIZE
    mask = col_offsets < n_cols
    
    # Load the row data into SRAM
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Subtract the maximum value in the row for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    
    # Compute the exponentials of the adjusted row values
    numerator = tl.exp(row_minus_max)
    
    # Sum the exponentials to get the denominator
    denominator = tl.sum(numerator, axis=0)
    
    # Compute the softmax output
    softmax_output = numerator / denominator
    
    # Compute the starting pointer for the output row
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Compute the output pointers for the current row
    output_ptrs = output_row_start_ptr + col_offsets
    
    # Store the softmax output back to the output tensor
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x):
    # Get the dimensions of the input tensor
    n_rows, n_cols = x.shape
    
    # Compute the block size as the next power of two of the number of columns
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Set the number of warps for the kernel execution
    num_warps = 8
    
    # Create an empty output tensor
    y = torch.empty_like(x)
    
    # Define the grid size for the kernel execution
    grid = (n_rows,)
    
    # Launch the softmax kernel
    softmax_kernel[grid](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )
    
    return y

# Example usage
torch.manual_seed(42)
x = torch.randn(1823, 781, device='cuda')
y_triton = softmax(x)
print(y_triton)
