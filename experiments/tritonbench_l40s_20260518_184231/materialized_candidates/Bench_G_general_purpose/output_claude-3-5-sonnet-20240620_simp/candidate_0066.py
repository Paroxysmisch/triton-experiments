import torch
import triton
import triton.language as tl

# Define block size for parallel processing
BLOCK_SIZE = 1024

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input vector
    y_ptr,  # Pointer to second input vector
    output_ptr,  # Pointer to output vector
    n_elements,  # Number of elements in the vectors
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this program instance
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where array size is not multiple of BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load data using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform addition
    output = x + y
    
    # Store the result using the same mask
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Input validation
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.shape == y.shape, "Input shapes must match"
    assert x.dtype == y.dtype, "Input types must match"
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate number of elements
    n_elements = output.numel()
    
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
