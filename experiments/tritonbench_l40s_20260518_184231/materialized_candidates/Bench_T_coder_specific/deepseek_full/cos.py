import torch
import triton
import triton.language as tl

@triton.jit
def cos(x):
    # Triton kernel to compute the cosine of a number
    return tl.cos(x)

def triton_cos(input, *, out=None):
    # Wrapper function for the Triton cosine function
    if out is None:
        out = torch.empty_like(input)
    assert out.is_contiguous()
    assert input.is_contiguous()
    grid = lambda meta: (triton.cdiv(input.numel(), meta['BLOCK_SIZE']),)
    cos[grid](input, out)
    return out
