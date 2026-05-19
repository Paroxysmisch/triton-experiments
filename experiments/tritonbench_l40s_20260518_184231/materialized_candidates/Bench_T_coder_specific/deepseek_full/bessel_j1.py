import torch
import triton
import triton.language as tl

@triton.jit
def _bessel_j1(x):
    # Bessel function of the first kind of order 1
    return x * tl.sin(x) / tl.cos(x / 2) ** 2

def bessel_j1(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.double)
    else:
        assert out.is_floating_point()
        assert out.shape == input.shape
    dim = input.ndim - 1
    input = input.contiguous()
    out = out.contiguous()

    def grid(meta):
        return (triton.cdiv(1, meta["BLOCK_SIZE"]),)

    _bessel_j1[grid](input, out=out, BLOCK_SIZE=1024, num_warps=1, num_stages=2, num_ctas=1)
    return out
