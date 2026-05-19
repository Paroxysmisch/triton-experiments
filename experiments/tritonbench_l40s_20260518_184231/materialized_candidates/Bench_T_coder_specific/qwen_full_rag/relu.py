import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(input, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input + offsets, mask=mask)
    result = tl.where(x >= 0, x, 0)
    tl.store(input + offsets, result, mask=mask)


def relu(input, inplace=False):
    if not inplace:
        input = input.clone()
    assert input.is_contiguous()
    N = input.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    relu_kernel[grid](input, N, BLOCK_SIZE=1024)
    return input
