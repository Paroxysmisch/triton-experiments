import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def _polygamma_jit(n, input, out, N, BLOCK_N: tl.constexpr):
    # Triton kernel implementation for polygamma
    pid = tl.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N

    inp = tl.load(input + offset, mask=mask)
    out_ = tl.digamma(inp)

    for i in range(1, n):
        out_ = tl.polygamma(inp, out_)

    tl.store(out + offset, out_, mask=mask)

def polygamma(n: int, input: Tensor, *, out: Tensor = None) -> Tensor:
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a nonnegative integer, but got {n}")

    if out is None:
        out = input.clone()
    elif out.is_floating_point():
        out = out.to(input.dtype)
    else:
        raise TypeError("out must be None or a tensor of floating type")

    N = input.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)
    _polygamma_jit[grid](n, input, out, N, BLOCK_N=4096)

    return out
