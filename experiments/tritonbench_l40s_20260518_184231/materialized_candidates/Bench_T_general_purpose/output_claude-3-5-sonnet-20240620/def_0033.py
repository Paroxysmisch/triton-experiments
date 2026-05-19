import triton
import triton.language as tl

@triton.jit
def logsumexp_kernel(input_ptr, output_ptr, dim, n_elements):
    # Get the thread's unique ID
    idx = tl.program_id(0)
    
    # Allocate shared memory for the sum of exponentials
    exp_sum = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over the input elements
    for i in range(n_elements):
        # Load the input value
        x = tl.load(input_ptr + idx * n_elements + i)
        # Update the sum of exponentials
        exp_sum[0] += tl.exp(x)
    
    # Compute the log of the sum of exponentials
    tl.store(output_ptr + idx, tl.log(exp_sum[0]))

import torch
from torch import Tensor

def logsumexp(input: Tensor, dim: int, keepdim: bool = False, *, out: Tensor = None) -> Tensor:
    # Ensure input is a tensor
    if not isinstance(input, Tensor):
        raise TypeError("Input must be a Tensor")
    
    # Get the shape of the input tensor
    n_elements = input.size(dim)
    
    # Prepare output tensor
    output = out if out is not None else input.new_zeros(input.size())
    
    # Launch the Triton kernel
    logsumexp_kernel[(input.size(0),)](input, output, dim, n_elements)
    
    # Handle the keepdim argument
    if keepdim:
        return output.unsqueeze(dim)
    return output.squeeze(dim)
