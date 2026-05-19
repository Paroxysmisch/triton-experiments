import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def mul_relu_kernel(input, other, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input + offsets, mask=mask)
    y = tl.load(other + offsets, mask=mask)
    xy = x * y
    out = tl.where(xy >= 0, xy, 0.0)
    tl.store(output + offsets, out, mask=mask)

def mul_relu(input, other, inplace=False, out=None) -> Tensor:
    check(
        len(input.shape) == 1,
        "mul_relu only accept 1D tensor now",
    )

    if inplace:
        check(
            out is None and input.is_contiguous(),
            "inplace mul_relu requires input to be contiguous",
        )
        mul_relu_kernel[(input.numel(),)](
            input,
            other,
            input,
            input.numel(),
            BLOCK_SIZE=TRITON_BLOCK_SIZE,
        )
        return input
    else:
        check(
            out is not None and not out.is_floating_point(),
            "out must be specified and must be integer type when inplace is False",
        )
        ret = torch.empty_like(input, dtype=torch.int64)
        mul_relu_kernel[(input.numel(),)](
            input, other, ret, input.numel(), BLOCK_SIZE=TRITON_BLOCK_SIZE
        )
        return ret
