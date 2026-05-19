import triton
import triton.language as tl

@triton.jit
def exp_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define the block index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Define the block end
    block_end = tl.min(block_start + BLOCK_SIZE, n_elements)
    
    # Iterate over the block range
    for i in range(block_start, block_end):
        # Load the input element
        x = tl.load(input_ptr + i)
        # Compute the exponential
        y = tl.exp(x)
        # Store the result in the output
        tl.store(output_ptr + i, y)


import torch

def exp(input, *, out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    # If no output tensor is provided, create one
    if out is None:
        out = torch.empty_like(input)
    
    # Check that the output tensor is the same size as the input tensor
    if out.size() != input.size():
        raise ValueError("output tensor must have the same size as input tensor")
    
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    
    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # Define a block size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # Define the grid size
    
    exp_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE)
    
    return out
