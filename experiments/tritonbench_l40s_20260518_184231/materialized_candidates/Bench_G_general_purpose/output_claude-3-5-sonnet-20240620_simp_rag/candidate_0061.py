import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,      # Pointer to first input vector
    y_ptr,      # Pointer to second input vector
    output_ptr, # Pointer to output vector
    n_elements, # Size of the vector
    BLOCK_SIZE: tl.constexpr,  # Static block size for parallel processing
):
    # Get the program ID for the current thread block
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for each element in the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle edge cases where block size doesn't evenly divide input size
    mask = offsets < n_elements
    
    # Load input vectors using the calculated offsets and mask
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform element-wise addition
    output = x + y
    
    # Store result in output vector
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # Input validation
    assert x.is_cuda and y.is_cuda, "Input tensors must be on GPU"
    assert x.shape == y.shape, "Input tensors must have the same shape"
    
    # Create output tensor with same properties as input
    output = torch.empty_like(x)
    
    # Calculate total number of elements
    n_elements = output.numel()
    
    # Define grid size based on block size and number of elements
    # triton.cdiv performs ceiling division to ensure all elements are processed
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel with block size of 1024
    add_kernel[grid](
        x, 
        y, 
        output, 
        n_elements, 
        BLOCK_SIZE=1024
    )
    
    return output
