# triton_cos.py

import triton
import triton.language as tl

@triton.jit
def cos_kernel(input_ptr, output_ptr, n_elements):
    # Compute the cosine for each element in the input tensor
    idx = tl.program_id(0)
    if idx < n_elements:
        output_ptr[idx] = tl.cos(input_ptr[idx])

def cos(input, *, out=None):
    # Ensure the output tensor is created if not provided
    if out is None:
        out = input.new_zeros(input.shape, dtype=input.dtype)
    
    # Launch the Triton kernel
    n_elements = input.numel()
    grid = (n_elements,)
    cos_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements)
    
    return out
