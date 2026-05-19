import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def exp_sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask, dtype=DTYPE)
    result = tl.sqrt(tl.exp(input))
    tl.store(output_ptr + offsets, result, mask=mask, dtype=DTYPE)

def exp_sqrt(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not input.is_cuda:
        raise TypeError("Input tensor must be on CUDA device")
    input = input.contiguous()
    n_elements = input.numel()
    
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor has incorrect shape")
        if out.dtype != input.dtype:
            raise TypeError("Output tensor dtype does not match input")
        if not out.is_cuda:
            raise TypeError("Output tensor must be on CUDA device")
        out = out.contiguous()
    
    dtype = input.dtype
    if dtype == torch.float32:
        triton_dtype = tl.float32
    elif dtype == torch.float16:
        triton_dtype = tl.float16
    elif dtype == torch.bfloat16:
        triton_dtype = tl.bfloat16
    else:
        raise NotImplementedError(f"Unsupported data type: {dtype}")
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    exp_sqrt_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements,
                          BLOCK_SIZE=1024, DTYPE=triton_dtype)
    return out
