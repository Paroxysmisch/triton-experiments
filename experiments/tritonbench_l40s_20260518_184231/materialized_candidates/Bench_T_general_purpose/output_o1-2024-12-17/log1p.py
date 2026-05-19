import torch
import triton
import triton.language as tl

@triton.jit
def _log1p_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)
    # Compute log1p(x) = log(1 + x)
    y = tl.log(1.0 + x)
    tl.store(out_ptr + offsets, y, mask=mask)

def log1p(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    assert input.is_cuda and out.is_cuda, "Tensors must be on GPU"
    n_elements = input.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _log1p_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
