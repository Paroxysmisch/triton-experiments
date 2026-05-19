import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def _quantize_global(
    x_ptr,
    absmax_inv_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)
    quantized = tl.libdevice.llrint(127.0 * x * absmax_inv)
    tl.store(output_ptr + offsets, quantized, mask=mask)

def quantize_global(x: torch.Tensor):
    if x.numel() == 0:
        raise ValueError("Input tensor must not be empty.")
    absmax = torch.max(torch.abs(x))
    if absmax == 0:
        return torch.zeros_like(x, dtype=torch.int8), absmax
    absmax_inv = 1.0 / absmax
    absmax_inv_tensor = torch.tensor([absmax_inv], device=x.device, dtype=x.dtype)
    output = torch.empty_like(x, dtype=torch.int8)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _quantize_global[grid](x, absmax_inv_tensor, output, n_elements)
    return output, absmax
