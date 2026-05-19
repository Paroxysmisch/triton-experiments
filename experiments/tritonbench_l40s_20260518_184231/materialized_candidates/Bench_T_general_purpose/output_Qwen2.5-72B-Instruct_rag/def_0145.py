import triton
import triton.language as tl
import torch

# Triton kernel for computing the n-th derivative of the digamma function
@triton.jit
def _polygamma_kernel(n, INPUT, OUTPUT, input_stride, output_stride, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    input_offsets = INPUT + offsets * input_stride
    output_offsets = OUTPUT + offsets * output_stride

    x = tl.load(input_offsets, mask=mask, other=0.0)

    # Compute the n-th derivative of the digamma function
    result = tl.where(n == 0, tl.digamma(x), tl.polygamma(n, x))

    tl.store(output_offsets, result, mask=mask)

# Python wrapper function for the polygamma kernel
def polygamma(n, input, *, out=None) -> torch.Tensor:
    if not input.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device")

    if out is None:
        out = torch.empty_like(input)

    if n < 0:
        raise ValueError("n must be a nonnegative integer")

    size = input.numel()
    grid = (triton.cdiv(size, 1024),)
    _polygamma_kernel[grid](n, input, out, input.stride(0), out.stride(0), size, BLOCK_SIZE=1024)

    return out
