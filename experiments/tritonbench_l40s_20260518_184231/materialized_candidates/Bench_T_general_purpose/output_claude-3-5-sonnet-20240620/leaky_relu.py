# triton_leaky_relu.py

import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(input_ptr, output_ptr, negative_slope, n_elements):
    # Define the grid size
    pid = tl.program_id(0)
    # Calculate the index for the current thread
    index = pid * tl.num_warps() + tl.arange(0, tl.num_warps())
    # Load input tensor
    input_data = tl.load(input_ptr + index)
    
    # Apply Leaky ReLU activation function
    output_data = tl.where(input_data > 0, input_data, negative_slope * input_data)
    
    # Store the result in the output tensor
    tl.store(output_ptr + index, output_data)

def leaky_relu(input, negative_slope=0.01, inplace=False):
    # Get the shape of the input tensor
    n_elements = input.numel()
    
    # Create an output tensor
    output = input.clone() if not inplace else input
    
    # Launch the Triton kernel
    leaky_relu_kernel[(n_elements,)](input.data_ptr(), output.data_ptr(), negative_slope, n_elements)
    
    return output
