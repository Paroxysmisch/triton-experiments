import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def exp_sqrt(input, out):
    # Triton kernel to compute exp followed by sqrt
    idx = tl.arange(0, 1)
    x = tl.load(input + idx)
    y = tl.exp(x)
    z = tl.sqrt(y)
    tl.store(out + idx, z)

def exp_sqrt(input: Tensor, out: Tensor = None) -> Tensor:
    # Wrapper function for Triton kernel
    if out is None:
        out = torch.empty_like(input)
    exp_sqrt[(1,)](input, out)
    return out
