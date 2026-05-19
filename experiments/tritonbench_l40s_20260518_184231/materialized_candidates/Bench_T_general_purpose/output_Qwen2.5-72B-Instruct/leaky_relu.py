import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(
    x_ptr,  # Pointer to the input tensor
    y_ptr,  # Pointer to the output tensor
    negative_slope,  # The negative slope
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute Leaky ReLU
    y = tl.where(x > 0, x, x * negative_slope)
    
    tl.store(y_ptr + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

def leaky_relu(input, negative_slope=0.01, inplace=False):
    if not isinstance(input, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    if not (0 <= negative_slope < 1):
        raise ValueError("negative_slope must be in the range [0, 1)")
    
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    
    leaky_relu_kernel[grid](
        input,  # Pointer to the input tensor
        output,  # Pointer to the output tensor
        negative_slope,  # The negative slope
        n_elements,  # Number of elements in the tensor
        BLOCK_SIZE=1024,  # Block size for parallelization
    )
    
    return output
