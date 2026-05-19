import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(input_ptr, output_ptr, negative_slope, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the block index
    pid = tl.program_id(0)
    # Compute the start and end indices for this block
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    # Clip the end index to the number of elements
    block_end = tl.min(block_end, n_elements)
    
    # Create a range of indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask for valid indices
    mask = offsets < block_end

    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply the Leaky ReLU function
    result = tl.max(0, x) + negative_slope * tl.min(0, x)
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

import torch

def leaky_relu(input, negative_slope=0.01, inplace=False):
    # Ensure the input is a contiguous tensor
    input = input.contiguous()
    
    # Prepare the output tensor
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    
    # Define the block size for the Triton kernel
    BLOCK_SIZE = 1024  # Adjust this based on your hardware capabilities
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    leaky_relu_kernel[grid](input, output, negative_slope, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output
