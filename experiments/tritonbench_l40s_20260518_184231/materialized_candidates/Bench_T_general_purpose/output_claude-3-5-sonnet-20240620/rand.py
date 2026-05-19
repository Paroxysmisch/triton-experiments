import triton
import triton.language as tl

@triton.jit
def rand_kernel(output_ptr, size, num_elements, **kwargs):
    # Generate random numbers in the range [0, 1)
    idx = tl.program_id(0)
    if idx < num_elements:
        # Use a simple linear congruential generator for randomness
        random_value = tl.random.uniform(0.0, 1.0)
        tl.store(output_ptr + idx, random_value)

import torch

def rand(*size, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False):
    # Calculate the total number of elements
    num_elements = torch.prod(torch.tensor(size)).item()
    
    # Create the output tensor
    if out is None:
        out = torch.empty(size, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, pin_memory=pin_memory)
    
    # Launch the Triton kernel
    rand_kernel[(num_elements,)](out.data_ptr(), size, num_elements, generator=generator)
    
    return out
