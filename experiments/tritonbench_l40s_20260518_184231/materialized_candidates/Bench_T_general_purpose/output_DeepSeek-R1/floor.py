import torch
import triton
import triton.language as tl

@triton.jit
def floor_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_values = tl.load(input_ptr + offsets, mask=mask)
    output_values = tl.math.floor(input_values)
    tl.store(output_ptr + offsets, output_values, mask=mask)

def floor(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Handle integer dtypes by returning a copy
    if input.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
        if out is None:
            return input.clone()
        else:
            if out.shape != input.shape:
                raise ValueError("out tensor has incorrect shape")
            if out.dtype != input.dtype:
                raise TypeError("out tensor has incorrect dtype")
            out.copy_(input)
            return out
    # For floating-point dtypes, compute floor
    else:
        input_contiguous = input.contiguous()
        if out is None:
            out = torch.empty_like(input_contiguous)
        else:
            if out.shape != input_contiguous.shape:
                raise ValueError("out tensor has incorrect shape")
            if out.dtype != input_contiguous.dtype:
                raise TypeError("out tensor has incorrect dtype")
            if not out.is_contiguous():
                raise ValueError("out tensor must be contiguous")
        n_elements = input_contiguous.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        floor_kernel[grid](input_contiguous.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=1024)
        return out
