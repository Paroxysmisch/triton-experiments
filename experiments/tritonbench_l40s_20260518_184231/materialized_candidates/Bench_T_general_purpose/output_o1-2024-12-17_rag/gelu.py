import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh, pow

@triton.jit
def gelu_none_kernel(x):
    x_fp32 = x.to(tl.float32)
    return 0.5 * x_fp32 * (1.0 + erf(0.7071067811 * x_fp32))

@triton.jit
def gelu_tanh_kernel(x):
    x_fp32 = x.to(tl.float32)
    return 0.5 * x_fp32 * (
        1.0 + tanh(0.7978845608 * (x_fp32 + 0.044715 * pow(x_fp32, 3)))
    )

class GeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, approximate='none'):
        if approximate == 'none':
            return gelu_none_kernel(input)
        elif approximate == 'tanh':
            return gelu_tanh_kernel(input)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")

def gelu(input, approximate='none'):
    return GeluFunction.apply(input, approximate)
