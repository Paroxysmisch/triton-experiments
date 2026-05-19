import torch
import triton
import triton.language as tl
from triton.language.math import erf, pow, tanh

@triton.jit
def gelu_none_and_mul_kernel(x, y):
    # Convert input to float32 for better precision in operations
    x_fp32 = x.to(tl.float32)
    # Compute the GELU function using the error function approximation
    x_gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))
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
            + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32.to(tl.float32), 2)))
        )
    )
    # Multiply the result by y and return
    return x_gelu * y

class GeluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, approximate="none"):
        # Log debug information
        logging.debug("GEMS GELU AND MUL FORWARD")
        # Choose the kernel based on the approximation method
        if approximate == "none":
            return gelu_none_and_mul_kernel(A, B)
        elif approximate == "tanh":
            return gelu_tanh_and_mul_kernel(A, B)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")

def gelu_and_mul(A, B, approximate="none"):
    # Wrapper function for using GeluAndMul class
    return GeluAndMul.apply(A, B, approximate)
