import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    # Pointers to matrices
    x_ptr,  # pointer to first input
    y_ptr,  # pointer to second input
    output_ptr,  # pointer to output
    n_elements,  # number of elements in the tensors
    BLOCK_SIZE: tl.constexpr,  # block size (will be specified at compile time)
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the block start and end
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements)
    
    # Create offsets for this program
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where block_size doesn't divide n_elements
    mask = offsets < n_elements
    
    # Load data using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform element-wise addition
    output = x + y
    
    # Store the result using the same mask
    tl.store(output_ptr + offsets, output, mask=mask)

def add_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Make sure input tensors are contiguous and on GPU
    x = x.contiguous()
    y = y.contiguous()
    
    # Check input dimensions
    assert x.is_cuda and y.is_cuda, "Input tensors must be on GPU"
    assert x.shape == y.shape, "Input tensors must have the same shape"
    
    # Get number of elements
    n_elements = x.numel()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Define block size (can be tuned for better performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    add_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
