import triton
import triton.language as tl

# Triton kernel to compute the element-wise square of each element in a 2D input tensor
@triton.jit
def square_kernel(
    x_ptr: tl.tensor,  # Input tensor
    y_ptr: tl.tensor,  # Output tensor
    x_row_stride: tl.int32,  # Row stride of the input tensor
    y_row_stride: tl.int32,  # Row stride of the output tensor
    n_cols: tl.int32,  # Number of columns in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size (power of two greater than number of columns)
):
    # Each block will handle one row
    row = tl.program_id(0)
    
    # Compute starting pointers for the row
    x_row_ptr = x_ptr + row * x_row_stride
    y_row_ptr = y_ptr + row * y_row_stride
    
    # Load the row into SRAM with masking
    x_row = tl.load(x_row_ptr, mask=tl.arange(BLOCK_SIZE) < n_cols)
    
    # Compute the square of each element in the row
    square_output = x_row * x_row
    
    # Store the squared values back to global memory
    tl.store(y_row_ptr, square_output, mask=tl.arange(BLOCK_SIZE) < n_cols)

# Wrapper function to compute the element-wise square of a 2D tensor
def square(x):
    import torch
    
    # Get the number of rows and columns
    n_rows, n_cols = x.shape
    
    # Determine BLOCK_SIZE as the smallest power of two greater than n_cols
    BLOCK_SIZE = 2 ** (n_cols - 1).bit_length()
    
    # Set the number of warps for parallel execution
    num_warps = min(BLOCK_SIZE // 32, 8)
    
    # Allocate an output tensor
    y = torch.empty_like(x)
    
    # Launch the kernel with a 1D grid configuration
    square_kernel[tl.arange(n_rows), BLOCK_SIZE, num_warps](x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE)
    
    return y
