import triton
import triton.language as tl
import torch

@triton.jit
def relu_kernel(
    x_ptr,    # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the start offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create an offset array for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements (handling the last block)
    mask = offsets < n_elements
    
    # Load input values using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Apply ReLU: max(0, x)
    output = tl.maximum(0, x)
    
    # Store results back to memory
    tl.store(out_ptr + offsets, output, mask=mask)

# Python wrapper for the Triton kernel
def relu_triton(x):
    # Get input size and create output tensor
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size (number of blocks)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    relu_kernel[grid](
        x_ptr=x,
        out_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
