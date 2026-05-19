import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn import SELU

@triton.jit
def selu_kernel(input, alpha, scale, **meta):
    idx = tl.arange(0, meta["N"])
    x = tl.load(input + idx).to(tl.float32)
    y = scale * tl.where(x >= 0, x, alpha * (tl.exp(x) - 1))
    tl.store(input + idx, y.to(meta["out_dtype"]))

def selu(input: Tensor, *, inplace: bool = False) -> Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if not inplace:
        input = input.clone()
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    with torch.cuda.device(input.device):
        selu_kernel[(input.numel(),)](input, alpha, scale, out_dtype=input.dtype)
    return input
