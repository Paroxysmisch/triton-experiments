# triton_signbit.py

import triton
import triton.language as tl

@triton.jit
def signbit_kernel(input_ptr, output_ptr, n_elements):
    # Iterate over each element in the input tensor
    for i in range(n_elements):
        # Load the input value
        value = tl.load(input_ptr + i)
        # Check if the sign bit is set (negative zero returns True)
        output_value = (value < 0) | (value == 0 and tl.signbit(value))
        # Store the result in the output tensor
        tl.store(output_ptr + i, output_value)

def signbit(input, *, out=None):
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    
    # Create output tensor if not provided
    if out is None:
        out = input.new_zeros(input.shape, dtype=tl.bool)
    
    # Launch the Triton kernel
    signbit_kernel[(n_elements,)](input.data_ptr(), out.data_ptr(), n_elements)
    
    return out
