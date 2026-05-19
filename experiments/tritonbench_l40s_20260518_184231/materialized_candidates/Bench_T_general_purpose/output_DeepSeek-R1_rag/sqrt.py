import triton
import triton.language as tl
import torch

@triton.jit
def sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_values = tl.load(input_ptr + offsets, mask=mask)
    output_values = tl.sqrt(input_values)
    tl.store(output_ptr + offsets, output_values, mask=mask)

def sqrt(input, *, out=None):
    if out is not None:
        if not out.is_floating_point():
            raise RuntimeError("out tensor must be floating point")
        input = input.to(out.dtype)
    else:
        if input.is_floating_point():
            out_dtype = input.dtype
        else:
            out_dtype = torch.float32
        out = torch.empty_like(input, dtype=out_dtype)
        input = input.to(out_dtype)
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024  # Fixed block size for efficient execution
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    sqrt_kernel[(grid_size, 1, 1)](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
