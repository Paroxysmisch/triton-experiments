import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Tuple

@triton.jit
def cos_signbit_kernel(input):
    # Compute the cosine of each element
    cos_result = tl.cos(input)
    # Determine the sign bit of each cosine result
    sign_bit = tl.signbit(cos_result)
    return cos_result, sign_bit

def cos_signbit(input: Tensor) -> Tuple[Tensor, Tensor]:
    # Ensure the input tensor is of float type
    assert input.dtype == torch.float32 or input.dtype == torch.float64 or input.dtype == torch.float16
    # Call the Triton kernel
    cos_result, sign_bit = cos_signbit_kernel(input)
    return cos_result, sign_bit
