import torch
import triton
import triton.language as tl
import math

@triton.jit
def cos_signbit_kernel(
    input_ptr,
    cos_output_ptr,
    sign_output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    input_values = tl.load(input_ptr + offset, mask=mask)
    input_float = input_values.to(tl.float32)
    cos_values = tl.cos(input_float)
    sign_bits = cos_values < 0.0
    tl.store(cos_output_ptr + offset, cos_values, mask=mask)
    tl.store(sign_output_ptr + offset, sign_bits, mask=mask)

def cos_signbit(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos_output = torch.empty_like(input)
    sign_output = torch.empty_like(input, dtype=torch.bool)
    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    cos_signbit_kernel[(grid_size, 1, 1)](
        input, cos_output, sign_output, n_elements, BLOCK_SIZE=block_size
    )
    return (cos_output, sign_output)
