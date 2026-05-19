# gelu.py
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(input_ptr, output_ptr, approximate, n_elements):
    # Define the index for each element
    idx = tl.program_id(0)
    
    # Ensure we don't go out of bounds
    if idx >= n_elements:
        return

    # Load the input value
    x = tl.load(input_ptr + idx)

    if approximate == 'none':
        # Compute the exact GELU: GELU(x) = x * Φ(x)
        # Φ(x) can be approximated using the error function
        phi_x = 0.5 * (1 + tl.erf(x / tl.sqrt(2.0)))
        result = x * phi_x
    elif approximate == 'tanh':
        # Compute the approximate GELU: GELU(x) = 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x^3)))
        result = 0.5 * x * (1 + tl.tanh(tl.sqrt(2 / tl.pi) * (x + 0.044715 * x**3)))
    else:
        raise ValueError("Invalid value for 'approximate'. Use 'none' or 'tanh'.")

    # Store the result
    tl.store(output_ptr + idx, result)

def gelu(input_tensor, approximate='none'):
    # Get the number of elements in the input tensor
    n_elements = input_tensor.shape[0]

    # Allocate output tensor
    output_tensor = input_tensor.new_zeros(input_tensor.shape)

    # Launch the kernel
    gelu_kernel[(n_elements,)](input_tensor.data_ptr(), output_tensor.data_ptr(), approximate, n_elements)

    return output_tensor
