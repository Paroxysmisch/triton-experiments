import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=1),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _quantize_global(
    x_ptr, 
    absmax_inv_ptr, 
    output_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    absmax_inv = tl.load(absmax_inv_ptr)
    scaled = x * absmax_inv * 127.0
    q = tl.libdevice.llrint(scaled)
    tl.store(output_ptr + offsets, q.to(tl.int8), mask=mask)

def quantize_global(x: torch.Tensor):
    absmax = x.abs().max()
    absmax_inv = 1.0 / absmax if absmax != 0 else 0.0
    absmax_inv_tensor = torch.tensor([absmax_inv], dtype=x.dtype, device=x.device)
    output = torch.empty_like(x, dtype=torch.int8)
    n_elements = x.numel()

    def grid(meta):
        return ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _quantize_global[grid](
        x,
        absmax_inv_tensor,
        output,
        n_elements
    )
    return output, absmax.item()
