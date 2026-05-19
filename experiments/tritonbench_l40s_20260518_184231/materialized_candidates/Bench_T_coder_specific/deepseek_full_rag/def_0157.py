import torch
import triton
import triton.language as tl
from typing import Tuple

# Triton kernel to compute sign bit check
@triton.jit
def signbit(input):
    return input.to(tl.float32) < 0

# Triton kernel to compute bitwise AND
@triton.jit
def bitwise_and(input, other):
    return input & other

# Wrapper function to call the Triton kernels
def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert other.is_integral() or other.is_boolean(), "The second tensor should be of integral or boolean types."
    signbit_result = signbit(input)
    bitwise_and_result = bitwise_and(input, other)
    return signbit_result, bitwise_and_result
