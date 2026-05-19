import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Tuple

@triton.jit
def cos_signbit_kernel(input, cos_result, signbit):
    # Convert input to float32
    input_fp32 = input.to(tl.float32)
    # Compute cosine of the input
    cos_result_fp32 = tl.cos(input_fp32)
    # Store the cosine result
    tl.store(cos_result, cos_result_fp32)
    # Compute the sign bit of the cosine result
    signbit_result = tl.signbit(cos_result_fp32)
    # Store the sign bit result
    tl.store(signbit, signbit_result)

def cos_signbit(input: Tensor) -> Tuple[Tensor, Tensor]:
    # Ensure input is a float tensor
    if input.dtype.is_fp64():
        input = input.to(tl.float32)
    # Prepare output tensors
    cos_result = torch.empty(input.shape, device=input.device, dtype=torch.float32)
    signbit = torch.empty(input.shape, device=input.device, dtype=torch.bool)
    # Call the Triton kernel
    cos_signbit_kernel[(1,)](input, cos_result, signbit)
    return cos_result, signbit
