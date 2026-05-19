import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh, pow

@triton.jit
def gelu_none_kernel(x):
    x_fp32 = x.to(tl.float32)
    return 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))

@triton.jit
def gelu_tanh_kernel(x):
    x_fp32 = x.to(tl.float32)
    return 0.5 * x_fp32 * (
        1 + tanh(0.79788456 * x_fp32 * (1 + 0.044715 * pow(x_fp32, 2)))
    )

class _MinGeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, keepdim, approximate, out):
        if approximate == 'none':
            x_gelu = gelu_none_kernel(input)
        elif approximate == 'tanh':
            x_gelu = gelu_tanh_kernel(input)
        else:
            raise ValueError("approximate must be 'none' or 'tanh'")

        if dim is not None:
            values, indices = torch.min(x_gelu, dim=dim, keepdim=keepdim)
            return torch.return_types.min(values, indices)
        else:
            min_val = torch.min(x_gelu)
            if out is not None:
                out.copy_(min_val)
                return out
            return min_val

def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    """
    Computes GELU on the input tensor and returns the minimum value along the specified dim.
    If dim is None, returns the minimum value of all elements.
    If dim is specified, returns a namedtuple (values, indices).
    """
    return _MinGeluFunction.apply(input, dim, keepdim, approximate, out)
