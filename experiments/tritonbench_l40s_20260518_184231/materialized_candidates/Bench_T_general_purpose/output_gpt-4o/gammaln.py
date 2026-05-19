import triton
import triton.language as tl

@triton.jit
def gammaln_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program ID
    pid = tl.program_id(axis=0)
    
    # Compute the range of elements this program should handle
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Compute the natural logarithm of the absolute value of the gamma function
    result = tl.log(tl.abs(tl.gamma(input_data)))
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=offsets < n_elements)


import torch

def gammaln(input, *, out=None):
    # Ensure the input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure output is a tensor
    if not isinstance(out, torch.Tensor):
        raise TypeError("Output must be a torch.Tensor")
    
    # Ensure input and output have the same shape
    if input.shape != out.shape:
        raise ValueError("Input and output tensors must have the same shape")
    
    # Get number of elements
    n_elements = input.numel()
    
    # Define block size (this can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gammaln_kernel[grid](input_ptr=input.data_ptr(), 
                         output_ptr=out.data_ptr(), 
                         n_elements=n_elements, 
                         BLOCK_SIZE=BLOCK_SIZE)
    
    return out
