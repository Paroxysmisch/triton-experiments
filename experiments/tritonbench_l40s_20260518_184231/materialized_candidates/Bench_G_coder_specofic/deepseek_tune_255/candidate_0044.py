import triton
import triton.language as tl
import torch
from torch import Tensor
from torch._inductor.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.utils import maybe_profile

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    stride,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride
    row_end_ptr = row_start_ptr + N
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_pointers = row_start_ptr + col_offsets
    mask = col_offsets < N
    row = tl.load(input_pointers, mask=mask, other=float("-inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_pointers = output_ptr + row_idx * stride + col_offsets
    tl.store(output_pointers, softmax_output, mask=mask)

def softmax(x: Tensor, dim=-1, **kwargs):
    shape = x.shape
    if dim >= len(shape):
        dim -= len(shape)
    assert dim == len(shape) - 1
    n = shape[dim]
    output = torch.empty_like(x)
    BLOCK_SIZE = grid(n, pow2roundup(n if n <= 2047 else 2047))
    assert BLOCK_SIZE in {64, 128, 256, 512, 1024, 2048}, BLOCK_SIZE
    num_warps = 2 if BLOCK_SIZE <= 512 else 4
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    softmax_kernel[grid](x, output, x.stride(0), n, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return output

def pow2roundup(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n
