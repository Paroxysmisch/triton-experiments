import torch
import triton
import triton.language as tl
import logging
from triton.language.math import erf, pow, tanh

@triton.jit
def gelu_none_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))
    return x_gelu * y

@triton.jit
def gelu_tanh_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_gelu = (
        0.5
        * x_fp32
        * (
            1
            + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32, 2)))
        )
    )
    return x_gelu * y

class GeluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, approximate="none"):
        logging.debug("GEMS GELU AND MUL FORWARD")
        if approximate == "none":
            return gelu_none_and_mul_kernel(A, B)
        elif approximate == "tanh":
            return gelu_tanh_and_mul_kernel(A, B)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")

def gelu_and_mul(A, B, approximate="none"):
    return GeluAndMul.apply(A, B, approximate)

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    gelu_result = gelu_and_mul(input, torch.ones_like(input), approximate)
    std_result = torch.std(gelu_result, dim=dim, keepdim=keepdim, correction=correction)
    if out is not None:
        out.copy_(std_result)
        return out
    return std_result
