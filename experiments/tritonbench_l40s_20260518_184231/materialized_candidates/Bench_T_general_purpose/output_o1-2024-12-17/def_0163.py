import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _cos_signbit_kernel(
    input_ptr, 
    cos_output_ptr, 
    signbit_output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    cos_val = tl.cos(x)
    tl.store(cos_output_ptr + offsets, cos_val, mask=mask)
    sign_val = cos_val < 0
    tl.store(signbit_output_ptr + offsets, sign_val, mask=mask)

def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out_cos = torch.empty_like(input)
    out_signbit = torch.empty_like(input, dtype=torch.bool)
    n_elements = input.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _cos_signbit_kernel[grid](input, out_cos, out_signbit, n_elements, BLOCK_SIZE=1024)
    return out_cos, out_signbit
