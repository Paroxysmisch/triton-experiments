import triton
import triton.language as tl

@triton.jit
def gelu_none_and_mul_kernel(x, y):
    # Convert input to float32 for better precision in operations
    x_fp32 = x.to(tl.float32)
    # Compute the GELU function using the error function approximation
    x_gelu = 0.5 * x_fp32 * (1 + tl.erf(x_fp32 * 0.7071067811))
    # Multiply the result by y and return
    return x_gelu * y

@triton.jit
def gelu_tanh_and_mul_kernel(x, y):
    # Convert input to float32 for better precision in operations
    x_fp32 = x.to(tl.float32)
    # Compute the GELU function using the tanh approximation
    x_gelu = (
        0.5
        * x_fp32
        * (
            1
            + tl.tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * tl.pow(x_fp32, 2)))
        )
    )
    # Multiply the result by y and return
    return x_gelu * y
