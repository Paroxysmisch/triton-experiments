import torch
import triton
import triton.language as tl

@triton.jit
def _log_tanh_kernel(
    x_ptr, 
    y_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = tl.log(x)
    x = tl.tanh(x)
    tl.store(y_ptr + offsets, x, mask=mask)

def log_tanh(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _log_tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
