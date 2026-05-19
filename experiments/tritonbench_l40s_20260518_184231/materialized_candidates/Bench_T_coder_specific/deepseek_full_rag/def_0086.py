import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def log_tanh_kernel(input_ptr, output_ptr, n_elements,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    x = tl.math.log(x)
    x = tl.math.tanh(x)
    tl.store(output_ptr + offsets, x, mask=mask)


def log_tanh(input: Tensor, out: Tensor = None) -> Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, but got {type(input)}")
    if not all(i > 0 for i in input):
        raise ValueError("all input elements must be positive for the logarithm to be defined")
    if out is None:
        out = torch.empty_like(input)
    elif out.shape != input.shape:
        raise ValueError(f"shape of out must be the same as input, but got {out.shape} vs {input.shape}")
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    log_tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
