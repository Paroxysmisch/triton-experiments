import torch
import triton
import triton.language as tl

@triton.jit
def sine_kernel(
    x_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Calculate the absolute position of the block
    pid = tl.program_id(axis=0)
    
    # Calculate start index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where array size is not multiple of BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load data using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute sine
    output = tl.sin(x)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def call_kernel(x: torch.Tensor) -> torch.Tensor:
    # Make sure input is on GPU
    assert x.is_cuda, "Input tensor must be on GPU"
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate grid and block dimensions
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    
    # Launch kernel
    sine_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=x.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
