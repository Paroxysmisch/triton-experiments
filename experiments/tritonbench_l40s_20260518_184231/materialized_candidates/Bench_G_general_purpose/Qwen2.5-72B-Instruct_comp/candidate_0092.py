import triton
import triton.language as tl

@triton.jit
def puzzle1_kernel(
    x_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    N,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size (compile-time constant)
    value: tl.constexpr  # Constant value to add
):
    pid = tl.program_id(axis=0)  # Get the program's unique ID along the specified axis (axis=0)
    block_start = pid * BLOCK_SIZE  # Compute the starting index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Determine the positions within this block
    mask = offsets < N  # Ensure computations only occur within valid indices

    # Load data from x_ptr
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Perform the element-wise addition
    y = x + value
    
    # Store the result in output_ptr
    tl.store(output_ptr + offsets, y, mask=mask)

import torch

def puzzle1(x: torch.Tensor) -> torch.Tensor:
    # Initialize an output tensor with the same shape and type as x
    output = torch.empty_like(x)
    
    # Verify CUDA compatibility of input and output tensors
    assert x.is_cuda and output.is_cuda, "Input and output tensors must be on a CUDA device."
    
    # Calculate the total number of elements
    N = x.numel()
    
    # Define the grid size
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    # Invoke the kernel with specified block size and constant value (10)
    puzzle1_kernel[grid](x, output, N, BLOCK_SIZE=1024, value=10)
    
    # Return the output tensor that contains the modified data
    return output
