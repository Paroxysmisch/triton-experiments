import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh
from collections import namedtuple

@triton.jit
def gelu_none_kernel(x):
    x_fp32 = x.to(tl.float32)
    gelu_result = 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))
    return gelu_result

@triton.jit
def gelu_tanh_kernel(x):
    x_fp32 = x.to(tl.float32)
    gelu_result = 0.5 * x_fp32 * (1 + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32, 2))))
    return gelu_result

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    if approximate == 'none':
        gelu_fn = gelu_none_kernel
    elif approximate == 'tanh':
        gelu_fn = gelu_tanh_kernel
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")

    # Apply the GELU function using the selected method
    gelu_result = gelu_fn(input)

    # Compute the minimum value along the specified dimension
    if dim is not None:
        min_values, min_indices = torch.min(gelu_result, dim=dim, keepdim=keepdim)
        MinResult = namedtuple('MinResult', ['values', 'indices'])
        return MinResult(min_values, min_indices)
    else:
        min_value = torch.min(gelu_result)
        return min_value

# Example usage
x = torch.randn(10, 10, device='cuda')
result = gelu_min(x, approximate='tanh', dim=1)
print(result)
