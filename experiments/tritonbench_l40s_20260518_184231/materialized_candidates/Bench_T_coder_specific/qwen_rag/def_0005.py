import triton
import triton.language as tl
import torch

# Kernel function to apply ReLU and then compute the square root
@triton.jit
def relu_sqrt_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_value = tl.load(a_ptr + offset, mask=mask)
    
    # Apply ReLU
    relu_value = tl.maximum(a_value, 0)
    
    # Compute the square root of the loaded elements
    b_value = tl.sqrt(relu_value.to(tl.float32))
    
    # Store the result in output tensor 'b' with boundary mask
    tl.store(b_ptr + offset, b_value, mask=mask)
