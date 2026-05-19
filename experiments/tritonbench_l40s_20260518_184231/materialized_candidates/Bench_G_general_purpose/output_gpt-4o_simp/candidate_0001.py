import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def _dequantize_rowwise(x_ptr, state_x_ptr, output_ptr, num_cols, BLOCK_SIZE: tl.constexpr):
    # Get the program id
    pid = tl.program_id(0)
    
    # Calculate the row start index
    row_start = pid * BLOCK_SIZE
    
    # Create a range of column indices
    col_idx = tl.arange(0, BLOCK_SIZE)
    
    # Check if the column index is within bounds
    mask = col_idx < num_cols
    
    # Load the maximum value for the current row from state_x
    max_val = tl.load(state_x_ptr + pid)
    
    # Load the input values for the current row
    x = tl.load(x_ptr + row_start * num_cols + col_idx, mask=mask, other=0)
    
    # Define the inverse of 127
    inv_127 = 1.0 / 127.0
    
    # Perform the dequantization
    dequantized = x * max_val * inv_127
    
    # Store the result in the output tensor
    tl.store(output_ptr + row_start * num_cols + col_idx, dequantized, mask=mask)

# Define the wrapper function
def dequantize_rowwise(x, state_x, BLOCK_SIZE=128):
    # Ensure x and state_x are on the GPU
    x = x.cuda()
    state_x = state_x.cuda()
    
    # Get the shape of the input tensor
    num_rows, num_cols = x.shape
    
    # Calculate the nearest power of two for the number of columns
    nearest_pow2 = 1 << (num_cols - 1).bit_length()
    
    # Allocate the output tensor
    output = torch.empty_like(x, dtype=torch.float32)
    
    # Launch the Triton kernel
    grid = (num_rows,)
    _dequantize_rowwise[grid](x, state_x, output, num_cols, BLOCK_SIZE=nearest_pow2)
    
    return output

# Example usage
if __name__ == "__main__":
    # Create example input data
    x = torch.randint(0, 128, (10, 128), dtype=torch.int32)
    state_x = torch.rand(10, dtype=torch.float32)
    
    # Perform dequantization
    output = dequantize_rowwise(x, state_x)
    print(output)
