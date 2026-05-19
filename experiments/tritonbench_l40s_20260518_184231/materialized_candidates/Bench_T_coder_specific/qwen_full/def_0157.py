import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Tuple

@triton.jit
def _signbit_bitwise_and(input, other):
    return _signbit(input.to(tl.float32)), input.to(tl.int64) & other

def signbit_bitwise_and(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    if input.is_floating_point() and other.is_floating_point():
        raise RuntimeError("At least one of the inputs must be of integral type")
    if input.dtype == torch.bool:
        input = input.to(torch.int64)
    if other.dtype == torch.bool:
        other = other.to(torch.int64)
    return _signbit_bitwise_and(input, other)
