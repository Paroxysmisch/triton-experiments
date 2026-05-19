import torch
import triton
import triton.language as tl
from triton.language.math import erf, pow, tanh
import logging


@triton.jit
def sub_gelu_exact_kernel(x, other, alpha):
    # Convert inputs to float32 for better precision
    x_fp32 = x.to(tl.float32)
    other_fp32 = other.to(tl.float32)
    # Subtract alpha * other from x and apply the exact GELU formula
    return x_fp32 * other_fp32 * (1 + erf((x_fp32 - alpha * other_fp32) * 0.7071067811))


@triton.jit
def sub_gelu_tanh_kernel(x, other, alpha):
    # Convert inputs to float32 for better precision
    x_fp32 = x.to(tl.float32)
    other_fp32 = other.to(tl.float32)
    # Subtract alpha * other from x and apply the tanh approximation of GELU
    return (
        0.5
        * x_fp32
        * other_fp32
        * (
            1
            + tanh(
                (x_fp32 - alpha * other_fp32)
                * 0.79788456
                * (1 + 0.044715 * pow(other_fp32.to(tl.float32), 2))
            )
        )
    )


class SubGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, alpha=1, approximate="none"):
        # Log debug information
        logging.debug("TRITON SUB GELU FORWARD")
        # Determine which kernel to use based on the approximate parameter
        if approximate == "none":
            return sub_gelu_exact_kernel(A, B, alpha)
        elif approximate == "tanh":
            return sub_gelu_tanh_kernel(A, B, alpha)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")


def sub_gelu(input, other, alpha=1, approximate="none", out=None) -> torch.Tensor:
    # Wrapper function for the SubGelu class
    return SubGelu.apply(input, other, alpha, approximate)
