import triton
import triton.language as tl

@triton.jit
def selu_kernel(
    x_ptr,  # pointer to the input tensor
    y_ptr,  # pointer to the output tensor
    n_elements,  # number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # block size for parallel processing
    alpha: tl.constexpr,  # SELU alpha constant
    scale: tl.constexpr  # SELU scale constant
):
    # Define the program index
    pid = tl.program_id(axis=0)
    
    # Define the range of indices this program will handle
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Apply the SELU function
    y = scale * (tl.max(x, 0.0) + tl.min(alpha * (tl.exp(x) - 1.0), 0.0))
    
    # Store the result
    tl.store(y_ptr + offsets, y, mask=offsets < n_elements)

import torch

def selu(input, inplace=False):
    # Define SELU constants
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    
    # Prepare input and output tensors
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    # Ensure the input is contiguous
    input = input.contiguous()
    
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    selu_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=1024,
        alpha=alpha,
        scale=scale
    )
    
    return output
