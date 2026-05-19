import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _erfc_sqrt_kernel(
    in_ptr, out_erfc_ptr, out_sqrt_ptr,
    n, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(in_ptr + offsets, mask=mask)
    y_erfc = tl.libdevice.erfc(x)
    y_sqrt = tl.sqrt(x)

    tl.store(out_erfc_ptr + offsets, y_erfc, mask=mask)
    tl.store(out_sqrt_ptr + offsets, y_sqrt, mask=mask)

def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.is_cuda, "Input must be a CUDA tensor."
    n = input.numel()
    out_erfc = torch.empty_like(input)
    out_sqrt = torch.empty_like(input)

    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _erfc_sqrt_kernel[grid](
        input, out_erfc, out_sqrt,
        n,
        BLOCK_SIZE=1024
    )
    return out_erfc, out_sqrt
