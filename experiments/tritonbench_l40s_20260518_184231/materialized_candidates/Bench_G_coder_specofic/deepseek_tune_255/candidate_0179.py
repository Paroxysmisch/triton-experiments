import triton
import triton.language as tl
import torch
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

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
    output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
    tl.store(output_ptr + offsets, output, mask=mask)

def quantize_global(x: Tensor) -> Tensor:
    absmax = x.abs().max().unsqueeze(0)
    absmax_inv = 1.0 / absmax
    output = torch.empty(*x.shape, device="cuda", dtype=torch.int8)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _quantize_global[grid](x, absmax_inv, output, n_elements)
    return output, absmax

@triton.jit
def _dequantize_global(
    x_ptr,
    absmax_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    absmax = tl.load(absmax_ptr)
    output = x * (1.0 / 127) * absmax
    tl.store(output_ptr + offsets, output, mask=mask)

def dequantize_global(x: Tensor, absmax: Tensor) -> Tensor:
    output = torch.empty(*x.shape, device="cuda", dtype=torch.float16)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _dequantize_global[grid](x, absmax, output, n_elements)
    return output
