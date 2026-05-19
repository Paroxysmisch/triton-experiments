import torch
import triton
import triton.language as tl
from typing import Tuple

# Triton kernel to compute the cosine of elements and determine their sign bits
@triton.jit
def _cos_signbit_kernel(
    x_ptr,
    y_cos_ptr,
    y_signbit_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute cosine
    x_cos = tl.cos(x.to(tl.float32))
    tl.store(y_cos_ptr + offsets, x_cos, mask=mask)
    
    # Determine sign bit
    x_signbit = x_cos < 0
    tl.store(y_signbit_ptr + offsets, x_signbit, mask=mask)

# Wrapper function to call the Triton kernel
def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    output_cos = torch.empty_like(input)
    output_signbit = torch.empty_like(input, dtype=torch.bool)
    
    assert (
        input.is_contiguous() and output_cos.is_contiguous()
    ), "Input and output buffers must be contiguous"
    
    n_elements = output_cos.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    _cos_signbit_kernel[grid](input, output_cos, output_signbit, n_elements, BLOCK_SIZE=1024)
    
    return output_cos, output_signbit
