import torch
import triton
import triton.language as tl

@triton.jit
def gammaln_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    input_abs = tl.abs(input)
    output = tl.math.lgamma(input_abs)
    tl.store(output_ptr + offsets, output, mask=mask)

def gammaln(input, *, out=None):
    input = input.contiguous()
    if not input.is_cuda:
        raise RuntimeError("gammaln requires CUDA tensors")
    if out is not None:
        if not out.is_cuda:
            raise RuntimeError("out must be a CUDA tensor")
        if out.shape != input.shape:
            raise RuntimeError("out shape must match input shape")
        if not out.is_contiguous():
            raise RuntimeError("out must be contiguous")
    else:
        out = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gammaln_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=1024)
    return out
