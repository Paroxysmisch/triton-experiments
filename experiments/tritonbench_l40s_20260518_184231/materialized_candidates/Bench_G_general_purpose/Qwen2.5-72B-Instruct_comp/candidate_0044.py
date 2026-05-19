import triton
import triton.language as tl

# Define the softmax kernel
@triton.jit
def softmax_kernel(input_ptr, output_ptr, stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Get the current row index
    row_idx = tl.program_id(0)
    
    # Calculate the input and output pointers for the row
    input_row_ptr = input_ptr + row_idx * stride
    output_row_ptr = output_ptr + row_idx * stride
    
    # Load the row data into SRAM
    row_data = tl.load(input_row_ptr, mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=-float('inf'))
    
    # Subtract the maximum value for numerical stability
    max_val = tl.max(row_data, axis=0)
    row_data = row_data - max_val
    
    # Compute the exponentials
    exp_row_data = tl.exp(row_data)
    
    # Compute the sum of the exponentials (denominator)
    sum_exp = tl.sum(exp_row_data, axis=0)
    
    # Compute the softmax values
    softmax_row_data = exp_row_data / sum_exp
    
    # Store the results back to the output tensor
    tl.store(output_row_ptr, softmax_row_data, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

# Define the softmax function
def softmax(input_tensor):
    # Get the input tensor dimensions
    n_rows, n_cols = input_tensor.shape
    
    # Create an empty output tensor
    output_tensor = tl.zeros_like(input_tensor)
    
    # Compute the block size as the next power of two of the number of columns
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE *= 2
    
    # Configure the grid and block sizes
    grid = (n_rows,)
    num_warps = 4 if BLOCK_SIZE <= 256 else 8
    
    # Invoke the softmax kernel
    softmax_kernel[grid](input_tensor, output_tensor, n_cols, BLOCK_SIZE, num_warps=num_warps)
    
    return output_tensor
