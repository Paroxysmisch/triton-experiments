import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Union
from collections import namedtuple

def gelu_min(
    input: Tensor,
    approximate: str = 'none',
    dim: Optional[int] = None,
    keepdim: bool = False,
    out: Optional[Tensor] = None
) -> Union[Tensor, namedtuple('Min', ['values', 'indices'])]:
    if dim is None:
        return triton.jit(gelu_min_nodim)(input).values
    else:
        Min = namedtuple('Min', ['values', 'indices'])
        values, indices = triton.jit(gelu_min_dim)(input, dim)
        return Min(values, indices)

@triton.jit
def gelu_min_nodim(input):
    input_f32 = input.to(tl.float32)
    if approximate == 'none':
        output = 0.5 * input_f32 * (1 + tl.erf(input_f32 / √2))
    else:
        output = 0.5 * input_f32 * (1 + tl.tanh(√2/π * (input_f32 + 0.044715 * input_f32 * input_f32 * input_f32)))
    min_val = tl.min(output, 0)
    return MinTensor(min_val, output)

@triton.jit
def gelu_min_dim(input, dim):
    input_f32 = input.to(tl.float32)
    if approximate == 'none':
        output = 0.5 * input_f32 * (1 + tl.erf(input_f32 / √2))
    else:
        output = 0.5 * input_f32 * (1 + tl.tanh(√2/π * (input_f32 + 0.044715 * input_f32 * input_f32 * input_f32)))
    output_shape = list(output.shape)
    output_shape[dim] = 1
    min_val = tl.min(output, dim).values
    indices = torch.argmin(output, dim, keepdim=keepdim)
    return MinTensor(min_val.to(output.dtype), indices.to(tl.int64))
