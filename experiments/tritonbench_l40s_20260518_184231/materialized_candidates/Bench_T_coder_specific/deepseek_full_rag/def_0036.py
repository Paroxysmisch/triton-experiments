import triton
import triton.language as tl

@triton.jit
def add_gelu_kernel(input, other, alpha, approximate):
    # Convert input to float32
    input_fp32 = input.to(tl.float32)
    # Add other to input
    result = input_fp32 + alpha * other
    # Apply GELU activation function
    if approximate == 'none':
        # Exact method using the Cumulative Distribution Function for Gaussian Distribution
        result = result * tl.math.erf(result * 0.7071067811)
    elif approximate == 'tanh':
        # Approximate method using a tanh-based formula
        result = result * 0.5 * (
            1 + tl.math.tanh(0.79788456 * result * (1 + 0.044715 * tl.math.pow(result, 2)))
        )
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")
    return result

def add_gelu(input, other, alpha=1, approximate='none', out=None):
    # Call the Triton kernel
    return add_gelu_kernel(input, other, alpha, approximate, out=out)
