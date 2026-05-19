import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh, pow

@triton.jit
def gelu_none_kernel(x):
    x_fp32 = x.to(tl.float32)
    # Exact GELU using erf
    return 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811865476))

@triton.jit
def gelu_tanh_kernel(x):
    x_fp32 = x.to(tl.float32)
    # Approximate GELU using tanh
    return 0.5 * x_fp32 * (
        1 + tanh(0.7978845608028654 * (x_fp32 + 0.044715 * pow(x_fp32, 3)))
    )

class GeluMin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, approximate='none', dim=None, keepdim=False, out=None):
        # Select the appropriate kernel based on the approximation argument
        if approximate == 'none':
            gelu_result = gelu_none_kernel(input)
        elif approximate == 'tanh':
            gelu_result = gelu_tanh_kernel(input)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")

        # Compute minimum along dim if provided, otherwise global minimum
        if dim is not None:
            values, indices = torch.min(gelu_result, dim=dim, keepdim=keepdim)
            if out is not None:
                out.copy_(values)
            ctx.save_for_backward(input)
            return (values, indices)
        else:
            min_val = torch.min(gelu_result)
            if out is not None:
                out.copy_(min_val)
            ctx.save_for_backward(input)
            return min_val

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    """
    gelu_min(input, approximate='none', dim=None, keepdim=False, out=None) -> Tensor or (Tensor, LongTensor)
    """
    return GeluMin.apply(input, approximate, dim, keepdim, out)
