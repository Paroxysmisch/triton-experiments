import torch
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import tag_sort

@triton.jit
def tanh(x):
    # Triton kernel for tanh
    return tl.libdevice.tanh(x)

@triton.jit
def tanh_kernel(input, out):
    # Triton kernel for tanh
    idx = tl.arange(0, 1)
    x = tl.load(input + idx)
    y = tanh(x)
    tl.store(out + idx, y)

def tanh(input, *, out=None):
    # Wrapper function for tanh
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32, device=input.device)
    assert input.is_contiguous()
    assert out.is_contiguous()
    assert input.dtype == torch.float32
    assert out.dtype == torch.float32
    assert input.device == out.device
    grid = (1,)
    tanh_kernel[grid](input, out)
    return out
