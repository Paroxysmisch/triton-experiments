import torch
import triton
import triton.language as tl

@triton.jit
def erf_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    output = tl.math.erf(x)
    tl.store(output_ptr + offsets, output, mask=mask)

def erf(input, *, out=None):
    if input.numel() == 0:
        return input if out is None else out
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("out tensor has incorrect shape")
        if out.dtype != input.dtype:
            raise ValueError("out tensor has incorrect dtype")
    n_elements = input.numel()
    input = input.contiguous()
    out = out.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    erf_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
