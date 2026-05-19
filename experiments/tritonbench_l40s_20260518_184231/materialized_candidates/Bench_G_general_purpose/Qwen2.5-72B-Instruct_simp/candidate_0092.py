import triton
import triton.language as tl

# Define the block size as a constant
BLOCK_SIZE = 128

# Triton kernel function
@triton.jit
def puzzle1_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the block index
    pid = tl.program_id(axis=0)
    # Compute the block start index
    block_start = pid * BLOCK_SIZE
    # Compute the offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load the input elements
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
    # Perform the element-wise addition
    y = x + 10
    # Store the output elements
    tl.store(output_ptr + offsets, y, mask=offsets < n_elements)

# Wrapper function to call the kernel
def puzzle1(x):
    # Get the number of elements in the input tensor
    n_elements = x.shape[0]
    # Allocate memory for the output tensor
    output = triton.empty_like(x)
    # Define the grid and block dimensions
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    # Launch the kernel
    puzzle1_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    return output
