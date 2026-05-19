import torch
import triton
import triton.language as tl

@triton.jit
def mul_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)

def mul(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    assert input.is_broadcastable_with(other), "The shapes of the `input` and `other` tensor must be broadcastable"
    in_casted = input.to(dtype=common_dtype(input, other))
    other_casted = other.to(dtype=common_dtype(input, other))
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    mul_kernel[grid](in_casted, other_casted, out, n_elements, BLOCK_SIZE=1024)
    return out
