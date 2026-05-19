import triton
import triton.language as tl
import torch

@triton.jit
def _sqrt_exp_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = tl.exp(tl.sqrt(x))
    tl.store(out_ptr + offsets, y, mask=mask)

def sqrt_exp(input, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    assert input.is_cuda, "Input tensor must be on CUDA."
    n_elements = input.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _sqrt_exp_kernel[grid](
        input.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE=1024
    )
    return out
