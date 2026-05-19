import triton
import triton.language as tl
import torch
import math

@triton.jit
def exp_sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_values = tl.load(input_ptr + offsets, mask=mask)
    # Compute exp and sqrt in float32 to handle precision and overflow
    input_f32 = input_values.to(tl.float32)
    exp_result = tl.exp(input_f32)
    sqrt_result = tl.sqrt(exp_result)
    tl.store(output_ptr + offsets, sqrt_result, mask=mask)

def exp_sqrt(input, out=None) -> torch.Tensor:
    input = input.contiguous()
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_contiguous(), "Output tensor must be contiguous"
        assert out.shape == input.shape, "Output shape must match input"
        assert out.dtype == input.dtype, "Output dtype must match input"
    n_elements = input.numel()
    # Calculate block size as the next power of two of the square root of elements
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    exp_sqrt_kernel[(grid_size, 1, 1)](input, out, n_elements, BLOCK_SIZE=block_size)
    return out
