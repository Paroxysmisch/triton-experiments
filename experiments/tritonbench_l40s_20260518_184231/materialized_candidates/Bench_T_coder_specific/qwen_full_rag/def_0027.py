import triton
import triton.language as tl
import torch
from torch import Tensor
from torch.autograd.function import Function
from torch.autograd import register_function

class SqrtTanh(Function):
    @staticmethod
    def forward(ctx, x, out=None):
        if out is None:
            out = torch.empty_like(x)
        else:
            assert x.is_contiguous() and out.is_contiguous()

        N = x.numel()
        BLOCK_SIZE = 1024
        grid_size = triton.cdiv(N, BLOCK_SIZE)

        _sqrt_tanh_kernel[grid_size](x, out, N, BLOCK_SIZE)
        return out

@register_function("sqrt_tanh")
def sqrt_tanh(A, out=None) -> Tensor:
    return SqrtTanh.apply(A, out)
