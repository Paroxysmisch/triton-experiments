import torch
import triton
import triton.language as tl
from triton.language import reduction
from triton.language.math import erf, tanh, pow

@triton.jit
def gelu_kernel(x, y):
    # Convert input to float32 for better precision in operations
    x_fp32 = x.to(tl.float32)
    # Compute the GELU function using the error function approximation
    x_gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))
    # Multiply the result by y and return
    return x_gelu * y

@triton.jit
def gelu_tanh_kernel(x, y):
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

@triton.jit
def min_kernel(x, dim):
    return reduction.min(x, dim)

class GeluAndMin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, approximate="none", dim=None):
        # Log debug information
        logging.debug("GEMS GELU AND MIN FORWARD")
        # Choose the kernel based on the approximation method
        if approximate == "none":
            gelu_result = gelu_kernel(A, B)
        elif approximate == "tanh":
            gelu_result = gelu_tanh_kernel(A, B)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")
        # Compute the minimum value along the specified dimension
        if dim is not None:
            min_result, indices = min_kernel(gelu_result, dim)
            return min_result, indices
        else:
            min_result = min_kernel(gelu_result)
            return min_result

def gelu_and_min(A, B, approximate="none", dim=None):
    # Wrapper function for using GeluAndMin class
    return GeluAndMin.apply(A, B, approximate, dim)
