import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import erf
from triton.language.math import pow, tanh

@triton.jit
def gelu_none_min_kernel(x, y):
    # Convert inputs to float32 for better precision
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    # Apply GELU using the error function approximation
    x_gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))
    # Multiply by y and compute the minimum
    return tl.minimum(x_gelu * y_fp32, y)

@triton.jit
def gelu_tanh_min_kernel(x, y):
    # Convert inputs to float32 for better precision
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    # Apply GELU using the tanh approximation
    x_gelu = (
        0.5
        * x_fp32
        * (
            1
            + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32.to(tl.float32), 2)))
        )
    )
    # Multiply by y and compute the minimum
    return tl.minimum(x_gelu * y_fp32, y)

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    # Wrapper function to apply GELU and find minimum
    if dim is None:
        input = input.contiguous()
        if approximate == 'none':
            output = gelu_none_min_kernel(input, input).to(input.dtype)
        else:
            output = gelu_tanh_min_kernel(input, input).to(input.dtype)
        if not keepdim:
            output = output.squeeze()
        return output
    else:
        if approximate == 'none':
            min_val, min_indices = tl.min(gelu_none_min_kernel(input, input), dim=dim, return_indices=True)
        else:
            min_val, min_indices = tl.min(gelu_tanh_min_kernel(input, input), dim=dim, return_indices=True)
        if out is not None:
            out = out.contiguous()
            min_val.copy_(out)
        if not keepdim:
            min_val = min_val.squeeze(dim=dim)
        return min_val, min_indices
