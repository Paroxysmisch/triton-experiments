import triton
import triton.language as tl

@triton.jit
def selu_kernel(input_ptr, output_ptr, scale, alpha, n_elements):
    # Get the index of the current element
    idx = tl.program_id(0)
    
    # Ensure we do not exceed the number of elements
    if idx >= n_elements:
        return
    
    # Load the input value
    x = tl.load(input_ptr + idx)

    # Compute SELU
    selu_value = scale * (tl.max(0, x) + tl.min(0, alpha * (tl.exp(x) - 1)))
    
    # Store the result
    tl.store(output_ptr + idx, selu_value)

def selu(input, inplace=False):
    # Constants for SELU
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    
    # Create an output tensor
    output = input.clone() if not inplace else input
    
    # Launch the kernel
    grid = (n_elements,)
    selu_kernel[grid](input.data_ptr(), output.data_ptr(), scale, alpha, n_elements)
    
    return output
