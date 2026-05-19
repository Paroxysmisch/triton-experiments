import torch
import triton
import triton.language as tl

@triton.jit
def floor(x):
    # Array API convention: for integer inputs, return a copy of the input tensor.
    if x.dtype.is_integer():
        return x
    else:
        return tl.math.floor(x)

def triton_floor(input: torch.Tensor, *, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    assert input.is_contiguous()
    assert out.is_contiguous()

    input_ = input if input.dtype.is_fp64() else input.to(torch.float32)
    out_ = out if out.dtype.is_fp64() else out.to(torch.float32)

    grid = lambda META: (triton.cdiv(input_.numel(), META['BLOCK_SIZE']),)
    floor[grid](input_, out=out_)

    return out_
