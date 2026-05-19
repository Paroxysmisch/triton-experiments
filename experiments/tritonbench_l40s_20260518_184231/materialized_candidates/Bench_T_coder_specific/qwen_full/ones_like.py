import torch
import triton
import triton.language as tl

@triton.jit
def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    return tl.full_like(input, 1, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)
