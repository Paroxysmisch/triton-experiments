import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def cos_signbit_kernel(
    input_ptr,
    cos_output_ptr,
    sign_output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_values = tl.load(input_ptr + offsets, mask=mask)
    cos_result = tl.cos(input_values)
    tl.store(cos_output_ptr + offsets, cos_result, mask=mask)

    cos_i32 = cos_result.view(tl.int32)
    sign_bits = (cos_i32 >> 31).to(tl.int1)
    tl.store(sign_output_ptr + offsets, sign_bits.to(tl.int8), mask=mask)

def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if not input.is_contiguous():
        input = input.contiguous()
    n_elements = input.numel()
    cos_output = torch.empty_like(input)
    sign_output = torch.empty_like(input, dtype=torch.bool)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    cos_signbit_kernel[grid](
        input, cos_output, sign_output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return (cos_output, sign_output)
