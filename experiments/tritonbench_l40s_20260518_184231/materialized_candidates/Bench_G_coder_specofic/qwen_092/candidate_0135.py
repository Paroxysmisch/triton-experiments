import triton
import triton.language as tl

# Define the kernel function
@triton.jit
def kernel_function(
    x_ptr: tl.tensor,  # Pointer to input data
    output_ptr: tl.tensor,  # Pointer to output data
    n_elements: tl.int32,  # Total number of elements to process
    BLOCK_SIZE: tl.constexpr  # Block size for block-wise operations
):
    # Calculate the block index and the linear thread index within the block
    block_idx = tl.program_id(0)
    thread_idx = tl.program_id(1)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    # Ensure we do not exceed the number of blocks
    if block_idx >= num_blocks:
        return

    # Calculate the starting point for the current block
    block_start = block_idx * BLOCK_SIZE
    # Calculate the starting point for the current thread
    thread_start = block_start + thread_idx

    # Ensure the thread index is within the valid range
    if thread_start >= n_elements:
        return

    # Create an offsets tensor for element indexing
    offsets = thread_start + tl.arange(0, BLOCK_SIZE)
    # Ensure offsets are within the valid range
    valid_mask = offsets < n_elements

    # Load data from x_ptr and apply the sine function
    x = tl.load(x_ptr + offsets, mask=valid_mask)
    output = tl.math.sin(x)

    # Store the result in output_ptr
    tl.store(output_ptr + offsets, output, mask=valid_mask)

# Define the wrapper function to call the kernel
def call_kernel(x):
    # Calculate the total number of elements
    n_elements = x.shape[0]

    # Create an output tensor with the same shape as the input tensor
    output = tl.zeros_like(x)

    # Define the block size
    BLOCK_SIZE = 256

    # Define the grid configuration function
    grid = lambda meta: (tl.cdiv(n_elements, BLOCK_SIZE), 1)

    # Launch the kernel
    kernel_function[grid](x, output, n_elements, BLOCK_SIZE)

    return output
