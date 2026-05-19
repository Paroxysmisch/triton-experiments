import torch
import triton
import triton.language as tl
from triton.language.libdevice import erf, exp, pow, tanh

@triton.jit
def add_gelu_no_approximation(x, y):
    # Convert inputs to float32 for better precision
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    # Add x and y
    x_add_y = x_fp32 + y_fp32
    # Calculate GELU using the exact method
    x_gelu = x_add_y * 0.5 * (1.0 + erf(x_add_y * 0.7071067811))
    return x_gelu

@triton.jit
def add_gelu_with_tanh_approximation(x, y):
    # Convert inputs to float32 for better precision
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    # Add x and y
    x_add_y = x_fp32 + y_fp32
    # Calculate GELU using the tanh approximation
    x_gelu = (
        0.5
        * x_add_y
        * (
            1.0
            + tanh(
                x_add_y * 0.79788456 * (1.0 + 0.044715 * pow(x_add_y.to(tl.float32), 2))
            )
        )
    )
    return x_gelu

class AddGELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, alpha, approximate):
        logging.debug("GEMS ADD GELU FORWARD")
        # Scale B by alpha
        B_scaled = alpha * B
        # Apply GELU with or without approximation
        if approximate == "none":
            return add_gelu_no_approximation(A, B_scaled)
        elif approximate == "tanh":
            return add_gelu_with_tanh_approximation(A, B_scaled)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")

def add_gelu(input, other, alpha=1, approximate="none", out=None) -> torch.Tensor:
    logging.debug("GEMS ADD GELU")
    # Wrapper function for the Triton kernel
    return AddGELUFunction.apply(input, other, alpha, approximate)
