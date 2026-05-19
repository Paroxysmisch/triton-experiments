import triton
import triton.language as tl
import torch
from typing import Union, Optional

@triton.jit
def pow_scalar_kernel(
    input_ptr,
    output_ptr,
    scalar_exponent,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    output_vals = tl.pow(input_vals, scalar_exponent)
    tl.store(output_ptr + offsets, output_vals, mask=mask)

@triton.jit
def pow_tensor_kernel(
    input_ptr,
    exponent_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    exponent_vals = tl.load(exponent_ptr + offsets, mask=mask)
    output_vals = tl.pow(input_vals, exponent_vals)
    tl.store(output_ptr + offsets, output_vals, mask=mask)

def pow(input: torch.Tensor, exponent: Union[float, torch.Tensor], *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if isinstance(exponent, (int, float)):
        if out is None:
            out = torch.empty_like(input)
        else:
            assert out.shape == input.shape, "Output tensor shape must match input"
        n_elements = input.numel()
        if n_elements == 0:
            return out
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        BLOCK_SIZE = 1024
        pow_scalar_kernel[grid](input, out, exponent, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out
    elif isinstance(exponent, torch.Tensor):
        try:
            input_bc, exponent_bc = torch.broadcast_tensors(input, exponent)
        except RuntimeError as e:
            raise RuntimeError(f"input and exponent shapes are not broadcastable: {input.shape} vs {exponent.shape}") from e
        input_bc = input_bc.contiguous()
        exponent_bc = exponent_bc.contiguous()
        if out is None:
            out = torch.empty_like(input_bc)
        else:
            assert out.shape == input_bc.shape, "Output tensor shape does not match broadcasted shape"
            out = out.contiguous()
        input_flat = input_bc.view(-1)
        exponent_flat = exponent_bc.view(-1)
        out_flat = out.view(-1)
        n_elements = input_flat.numel()
        if n_elements == 0:
            return out
        assert input_bc.is_cuda and exponent_bc.is_cuda and out.is_cuda, "Tensors must be on CUDA device"
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        BLOCK_SIZE = 1024
        pow_tensor_kernel[grid](input_flat, exponent_flat, out_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out
    else:
        raise TypeError("exponent must be a float or tensor")
