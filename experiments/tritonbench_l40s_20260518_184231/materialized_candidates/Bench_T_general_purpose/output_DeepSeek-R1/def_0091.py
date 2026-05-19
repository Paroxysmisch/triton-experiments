import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def erfc_sqrt_kernel(
    input_ptr,
    erfc_output_ptr,
    sqrt_output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    erfc_val = 1.0 - tl.math.erf(x)
    sqrt_val = tl.sqrt(x)
    tl.store(erfc_output_ptr + offsets, erfc_val, mask=mask)
    tl.store(sqrt_output_ptr + offsets, sqrt_val, mask=mask)

def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.is_contiguous(), "Input tensor must be contiguous"
    assert input.is_cuda, "Input tensor must be on CUDA device"
    
    erfc_out = torch.empty_like(input)
    sqrt_out = torch.empty_like(input)
    n_elements = input.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024  # Adjust based on optimal performance for the hardware
    
    erfc_sqrt_kernel[grid](
        input.data_ptr(),
        erfc_out.data_ptr(),
        sqrt_out.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (erfc_out, sqrt_out)
