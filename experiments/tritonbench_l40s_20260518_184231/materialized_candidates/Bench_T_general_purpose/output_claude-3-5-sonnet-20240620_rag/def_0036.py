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
    def forward(ctx, input, other, alpha=1, approximate="none"):
        # Log debug information
        # Compute the scaled addition
        scaled_other = alpha * other
        # Compute the result based on the approximation method
        if approximate == "none":
            return gelu_none_and_mul_kernel(input, scaled_other)
        elif approximate == "tanh":
            return gelu_tanh_and_mul_kernel(input, scaled_other)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")

def add_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    # Wrapper function for using GeluAndMul class
    return GeluAndMul.apply(input, other, alpha, approximate)
