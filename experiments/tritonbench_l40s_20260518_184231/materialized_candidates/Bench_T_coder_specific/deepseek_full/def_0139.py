import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def _std_impl(input, dim, correction, keepdim, out):
    input = input.to(tl.float32)
    mean = tl.sum(input, axis=dim) / tl.num_programs(dim)
    input_minus_mean = input - mean
    square = input_minus_mean * input_minus_mean
    sum_square = tl.sum(square, axis=dim) / tl.num_programs(dim)
    var = sum_square / (tl.full(tl.float32, input.numel(), 1.0 - correction) / correction)
    std = tl.sqrt(var)
    if keepdim:
        std = tl.expand_dims(std, dim)
    return std

def std(input, dim=None, *, correction=1, keepdim=False, out=None) -> Tensor:
    if isinstance(dim, int):
        dim = [dim]
    elif dim is None:
        dim = list(range(input.ndim))
    if out is None:
        out = torch.empty(size=[input.size(d) if keepdim else 1 for d in dim], dtype=input.dtype, device=input.device)
    else:
        assert out.dtype == input.dtype
        assert out.device == input.device
        assert out.shape == tuple([input.size(d) if keepdim else 1 for d in dim])
    _std_impl[input.numel(),](input, dim, correction, keepdim, out)
    return out
