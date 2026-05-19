import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def exp_mean_kernel(input, output, N, **meta):
    idx = tl.arange(0, N)
    x = tl.load(input + idx).to(tl.float32)
    y = tl.exp(x)
    mean = tl.sum(y) / N
    tl.store(output, mean)

def exp_mean(input: Tensor, dim=None, keepdim=False, dtype=None, out=None) -> Tensor:
    if out is None:
        out = torch.empty([], dtype=torch.float32, device=input.device)

    if dim is not None:
        input = input.unsqueeze(dim)
        keepdim = True

    input = input.contiguous()

    N = input.numel()
    grid = lambda meta: (1,)
    exp_mean_kernel[grid](input, out, N)

    if not keepdim:
        out.squeeze_()

    return out
