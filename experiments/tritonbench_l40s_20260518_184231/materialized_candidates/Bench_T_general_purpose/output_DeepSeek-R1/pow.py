import torch
import triton
import triton.language as tl
from typing import Optional, Union

@triton.jit
def pow_scalar_kernel(
    input_ptr,
    exponent_scalar,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_elements
    input = tl.load(input_ptr + idx, mask=mask)
    output = tl.pow(input, exponent_scalar)
    tl.store(output_ptr + idx, output, mask=mask)

@triton.jit
def pow_tensor_kernel(
    input_ptr,
    exponent_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_elements
    input = tl.load(input_ptr + idx, mask=mask)
    exponent = tl.load(exponent_ptr + idx, mask=mask)
    output = tl.pow(input, exponent)
    tl.store(output_ptr + idx, output, mask=mask)

def pow(input: torch.Tensor, exponent: Union[float, torch.Tensor], *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if isinstance(exponent, (float, int)):
        if out is None:
            out = torch.empty_like(input)
        num_elements = input.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
        pow_scalar_kernel[grid](
            input, exponent, out, num_elements,
            BLOCK_SIZE=1024
        )
        return out
    else:
        broadcasted_shape = torch.broadcast_shapes(input.shape, exponent.shape)
        input_expanded = input.expand(broadcasted_shape).contiguous()
        exponent_expanded = exponent.expand(broadcasted_shape).contiguous()
        if out is None:
            out = torch.empty_like(input_expanded)
        else:
            out = out.contiguous()
        input_flat = input_expanded.view(-1)
        exponent_flat = exponent_expanded.view(-1)
        out_flat = out.view(-1)
        num_elements = out_flat.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
        pow_tensor_kernel[grid](
            input_flat, exponent_flat, out_flat, num_elements,
            BLOCK_SIZE=1024
        )
        return out
