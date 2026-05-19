import torch
import triton
import triton.language as tl
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
            + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32.to(tl.float32), 2)))
        )
    )
    return x_gelu * y

class GeluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, approximate="none"):
        if approximate == "none":
            return gelu_none_and_mul_kernel(A, B)
        elif approximate == "tanh":
            return gelu_tanh_and_mul_kernel(A, B)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")

def gelu_and_mul(A, B, approximate="none"):
    return GeluAndMul.apply(A, B, approximate)

def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Shape of out and input must be the same"
    
    if isinstance(other, torch.Tensor):
        assert input.shape == other.shape, "Shape of input and other must be the same"
        other_sub = other
    else:
        other_sub = torch.full_like(input, other, dtype=input.dtype)
    
    input_sub = input - alpha * other_sub
    out_gelu = gelu_and_mul(input_sub, out, approximate)
    
    return out_gelu
