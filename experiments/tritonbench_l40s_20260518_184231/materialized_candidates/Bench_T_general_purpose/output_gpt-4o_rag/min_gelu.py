import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh, sqrt

@triton.jit
def gelu_none_kernel(x):
    x_fp32 = x.to(tl.float32)
    return 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))

@triton.jit
def gelu_tanh_kernel(x):
    x_fp32 = x.to(tl.float32)
    return 0.5 * x_fp32 * (1 + tanh(sqrt(2 / 3.141592653589793) * (x_fp32 + 0.044715 * x_fp32**3)))

class MinGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False, approximate='none'):
        # Choose the kernel based on the approximation method
        if approximate == 'none':
            gelu_func = gelu_none_kernel
        elif approximate == 'tanh':
            gelu_func = gelu_tanh_kernel
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")

        # Compute GELU activation
        gelu_output = gelu_func(input)

        # Find the minimum value along the specified dimension
        if dim is None:
            min_val, min_idx = gelu_output.min(), None
        else:
            min_val, min_idx = gelu_output.min(dim=dim, keepdim=keepdim)

        ctx.save_for_backward(input, min_idx)
        ctx.approximate = approximate
        ctx.dim = dim
        ctx.keepdim = keepdim

        return min_val

def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    return MinGELU.apply(input, dim, keepdim, approximate)

# Example usage:
# input_tensor = torch.randn(10, 10, device='cuda')
# result = min_gelu(input_tensor, dim=1, keepdim=True, approximate='tanh')
