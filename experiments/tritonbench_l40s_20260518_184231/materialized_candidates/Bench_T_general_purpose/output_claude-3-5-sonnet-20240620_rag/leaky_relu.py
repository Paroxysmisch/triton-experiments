import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(x, negative_slope, output):
    """
    Leaky ReLU activation function kernel.
    Applies the Leaky ReLU activation element-wise to the input tensor.
    """
    # Calculate the Leaky ReLU
    output[tl.program_id(0)] = tl.where(x >= 0, x, negative_slope * x)

def leaky_relu(input, negative_slope=0.01, inplace=False):
    """
    Wrapper function for the Leaky ReLU activation function.
    
    Parameters:
    - input: Input tensor.
    - negative_slope: Slope for negative input values (default: 0.01).
    - inplace: If True, modifies the input tensor directly (default: False).
    
    Returns:
    - Tensor after applying Leaky ReLU.
    """
    output = input if inplace else tl.empty_like(input)
    leaky_relu_kernel[(input.numel(),)](input, negative_slope, output)
    return output
