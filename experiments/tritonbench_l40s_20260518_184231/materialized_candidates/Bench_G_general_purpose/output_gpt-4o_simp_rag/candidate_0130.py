import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 128

# Triton kernel for element-wise addition
@triton.jit
def add_kernel(X, Y, Z, N, BLOCK_SIZE: tl.constexpr):
    # Calculate offsets for the block
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Ensure we don't go out of bounds
    mask = offsets < N
    
    # Load data from input tensors
    x_vals = tl.load(X + offsets, mask=mask, other=0.0)
    y_vals = tl.load(Y + offsets, mask=mask, other=0.0)
    
    # Perform element-wise addition
    z_vals = x_vals + y_vals
    
    # Store the result in the output tensor
    tl.store(Z + offsets, z_vals, mask=mask)

# Wrapper function to initialize output tensor and launch the kernel
def add_wrapper(x_ptr, y_ptr, z_ptr, num_elements):
    # Calculate number of blocks needed
    grid = lambda META: (num_elements + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE']
    
    # Launch the kernel
    add_kernel[grid](x_ptr, y_ptr, z_ptr, num_elements, BLOCK_SIZE=BLOCK_SIZE)

# Example usage
# Assuming you have pointers x_ptr, y_ptr, and z_ptr to your input and output tensors
# and num_elements representing the number of elements in the tensors:
# add_wrapper(x_ptr, y_ptr, z_ptr, num_elements)
