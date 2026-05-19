import triton
import triton.language as tl

@triton.jit
def sub_kernel(input_ptr, other_ptr, out_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program index for the current block
    pid = tl.program_id(0)
    
    # Create a block of indices
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load data from input and other, applying the mask
    input_data = tl.load(input_ptr + offsets, mask=mask)
    other_data = tl.load(other_ptr + offsets, mask=mask)
    
    # Perform the operation
    result = input_data - alpha * other_data
    
    # Store the result back to the output pointer
    tl.store(out_ptr + offsets, result, mask=mask)

import torch

def sub(input, other, *, alpha=1, out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a Tensor")
    
    # Convert other to a tensor if it is a number
    if isinstance(other, (int, float, complex)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Handle broadcasting and type promotion
    input, other = torch.broadcast_tensors(input, other)
    
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure the output tensor is the correct shape
    if out.shape != input.shape:
        raise ValueError("The output tensor has an incorrect shape.")
    
    # Get the number of elements
    n_elements = input.numel()
    
    # Allocate device memory for input, other, and output
    input_ptr = input.data_ptr()
    other_ptr = other.data_ptr()
    out_ptr = out.data_ptr()
    
    # Define block size for Triton kernel
    BLOCK_SIZE = 1024  # Example block size, adjust based on your hardware
    
    # Launch the Triton kernel
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    sub_kernel[grid](input_ptr, other_ptr, out_ptr, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
