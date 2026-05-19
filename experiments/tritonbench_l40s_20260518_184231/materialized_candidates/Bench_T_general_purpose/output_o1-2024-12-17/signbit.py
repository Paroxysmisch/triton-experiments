import torch
import triton
import triton.language as tl

@triton.jit
def _signbit_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)
    x_int = tl.bitcast(x, tl.int32)
    sign = (x_int >> 31) & 1

    tl.store(out_ptr + offsets, sign, mask=mask)

def signbit(input: torch.Tensor, *, out: torch.Tensor = None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)
    assert input.is_cuda and out.is_cuda, "Tensors must be CUDA tensors."
    n_elements = input.numel()
    grid = lambda meta: ( (n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    _signbit_kernel[grid](
        input, out, n_elements,
        BLOCK_SIZE=1024
    )
    return out
