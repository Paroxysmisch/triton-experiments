import triton
import triton.language as tl

# Define the softmax kernel
@triton.jit
def softmax_kernel(
    input_ptr: tl.tensor,  # Pointer to the input matrix
    output_ptr: tl.tensor, # Pointer to the output matrix
    input_row_stride: tl.int32, # Stride for row advancement in input matrix
    output_row_stride: tl.int32, # Stride for row advancement in output matrix
    n_cols: tl.int32, # Number of columns in the matrix
    BLOCK_SIZE: tl.constexpr  # Block size for parallel computation
):
    # Get the current row index
    row_id = tl.program_id(0)
    
    # Calculate the starting pointer for this row
    row_ptr = input_ptr + row_id * input_row_stride
    
    # Load the row into shared memory with masking
    row = tl.load(row_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(BLOCK_SIZE) < n_cols, boundary_check=False)
    
    # Find the maximum value in the row
    max_val = tl.max(row)
    
    # Subtract the maximum value for numerical stability
    row = row - max_val
    
    # Compute the exponentials
    exp_row = tl.exp(row)
    
    # Compute the sum of the exponentials
    exp_sum = tl.sum(exp_row)
    
    # Normalize to get the softmax probabilities
    softmax_row = exp_row / exp_sum
    
    # Store the result back in the output matrix
    tl.store(output_ptr + row_id * output_row_stride, softmax_row, mask=tl.arange(BLOCK_SIZE) < n_cols)

# Define the wrapper function for the softmax kernel
def softmax(input_tensor):
    # Get the number of columns
    n_cols = input_tensor.shape[1]
    
    # Find the smallest power of two greater than the column count
    BLOCK_SIZE = 2 ** tl.bitwise_ceil(tl.log2(n_cols))
    
    # Adjust num_warps to ensure efficient parallel execution
    num_warps = BLOCK_SIZE // 32
    
    # Create an output tensor with the same shape as the input tensor
    output_tensor = tl.zeros_like(input_tensor)
    
    # Launch the Triton kernel with one block per input matrix row
    grid = (input_tensor.shape[0], 1, 1)
    softmax_kernel[grid, (BLOCK_SIZE, 1, 1), (num_warps, 1, 1)](
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        input_tensor.stride(0),
        output_tensor.stride(0),
        n_cols,
        BLOCK_SIZE
    )
    
    return output_tensor
