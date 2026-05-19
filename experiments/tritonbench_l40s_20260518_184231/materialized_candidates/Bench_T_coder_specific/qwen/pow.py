import torch
import triton
import triton.language as tl

# Define the Triton kernel
pow_kernel = triton.compile(pow_kernel, constants={"block_size": 256})

def pow(input, exponent, out=None):
    # Get input and exponent shapes
    input_shape = input.shape
    exponent_shape = exponent.shape
    
    # Check if output is provided, otherwise create a new tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Flatten input and exponent tensors
    input_flat = input.flatten()
    exponent_flat = exponent.flatten()
    
    # Determine the number of elements
    num_elements = input_flat.numel()
    
    # Launch the Triton kernel
    grid_size = (num_elements + 255) // 256
    pow_kernel[(grid_size,)](input_flat.data_ptr(), 
                              exponent_flat.data_ptr(), 
                              out.data_ptr(),
                              input_shape,
                              exponent_shape,
                              num_elements)
    
    return out
