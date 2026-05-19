import triton
import triton.language as tl

# Define the BLOCK_SIZE
BLOCK_SIZE = 32  # Adjust this value based on your specific use case

@triton.jit
def softmax_kernel(
    X_ptr,  # input tensor
    Y_ptr,  # output tensor
    stride_x,  # stride for row access in X
    stride_y,  # stride for row access in Y
    BLOCK_SIZE: tl.constexpr  # block size for parallel computation
):
    # Get the current row index
    row = tl.program_id(0)
    x = tl.load(X_ptr + row * stride_x)
    
    # Find the maximum value in the row for numerical stability
    max_val = tl.max(x)
    
    # Subtract the maximum value from each element
    x = x - max_val
    
    # Compute the exponentials (numerator)
    exp_x = tl.exp(x)
    
    # Compute the sum of exponentials (denominator)
    exp_sum = tl.sum(exp_x)
    
    # Compute the softmax result
    y = exp_x / exp_sum
    
    # Store the result in the output tensor
    tl.store(Y_ptr + row * stride_y, y)

def softmax(x):
    # Get the shape of the input tensor
    n, d = x.shape
    
    # Compute the BLOCK_SIZE as the next power of two of the number of columns
    BLOCK_SIZE = 2 ** tl.bit_length(d - 1)
    
    # Create an empty output tensor
    y = tl.zeros_like(x)
    
    # Configure the grid and block sizes
    grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    block = BLOCK_SIZE
    
    # Launch the softmax kernel
    softmax_kernel[grid, block](
        x.data_ptr(),  # input tensor
        y.data_ptr(),  # output tensor
        x.stride(0),  # stride for row access in X
        y.stride(0),  # stride for row access in Y
        BLOCK_SIZE  # block size
    )
    
    return y
