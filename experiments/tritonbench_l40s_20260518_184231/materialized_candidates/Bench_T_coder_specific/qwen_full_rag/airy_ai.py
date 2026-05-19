import torch
import triton
import triton.language as tl

from ..special.airy import airy_ai as torch_airy_ai


@triton.jit
def airy_ai_kernel(x, out, SIZE: tl.constexpr):
    idx = tl.arange(0, SIZE)
    x_val = tl.load(x + idx)
    y_val = tl.math.airy_ai(x_val)
    tl.store(out + idx, y_val)


def airy_ai(input: torch.Tensor, *, out=None) -> torch.Tensor:
    input = input.contiguous()
    if out is None:
        out = torch.empty_like(input)
    else:
        out = out.contiguous()
    assert input.is_cuda and out.is_cuda
    N = input.numel()
    grid = lambda meta: (triton.cdiv(N, meta['SIZE']), )
    airy_ai_kernel[grid](input, out, SIZE=1024)
    return out
