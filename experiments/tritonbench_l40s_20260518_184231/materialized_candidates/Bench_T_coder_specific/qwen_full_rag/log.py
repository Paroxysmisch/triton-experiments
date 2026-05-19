import torch
import triton
import triton.language as tl

@triton.jit
def log_kernel(input, output, len, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < len
    x = tl.load(input + offset, mask=mask)
    y = tl.math.log(x)
    tl.store(output + offset, y, mask=mask)

def log(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    assert input.is_contiguous()
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    dtype = input.dtype
    if dtype == torch.float32:
        log_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    elif dtype == torch.float64:
        log_kernel[grid](input, out, n_elements, BLOCK_SIZE=512)
    else:
        raise RuntimeError("Only fp32/fp64 are supported")
    return out
