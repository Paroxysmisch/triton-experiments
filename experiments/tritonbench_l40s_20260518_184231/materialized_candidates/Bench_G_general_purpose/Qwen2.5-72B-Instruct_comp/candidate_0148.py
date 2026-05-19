import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 1024

# Kernel to multiply elements by 2 and store the result in a new tensor
@triton.jit
def mul2_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program ID
    pid = tl.program_id(0)
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    # Compute offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to ensure we stay within bounds
    mask = offsets < n_elements
    # Load the elements from memory
    input_block = tl.load(input_ptr + offsets, mask=mask)
    # Multiply the elements by 2
    output_block = input_block * 2
    # Store the result back to memory
    tl.store(output_ptr + offsets, output_block, mask=mask)

# Kernel to multiply elements by 2 in place
@triton.jit
def mul2_inplace_kernel(input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program ID
    pid = tl.program_id(0)
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    # Compute offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to ensure we stay within bounds
    mask = offsets < n_elements
    # Load the elements from memory
    input_block = tl.load(input_ptr + offsets, mask=mask)
    # Multiply the elements by 2
    output_block = input_block * 2
    # Store the result back to memory
    tl.store(input_ptr + offsets, output_block, mask=mask)

# Wrapper function to call the mul2_kernel
def triton_mul2(input_tensor, output_tensor):
    n_elements = input_tensor.shape[0]
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    mul2_kernel[grid_size, BLOCK_SIZE](input_tensor, output_tensor, n_elements, BLOCK_SIZE)

# Wrapper function to call the mul2_inplace_kernel
def triton_mul2_inplace(input_tensor):
    n_elements = input_tensor.shape[0]
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    mul2_inplace_kernel[grid_size, BLOCK_SIZE](input_tensor, n_elements, BLOCK_SIZE)
