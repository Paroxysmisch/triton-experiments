import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def bitwise_and_kernel(
    input_ptr, other_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)
    output_vals = input_vals & other_vals
    tl.store(output_ptr + offsets, output_vals, mask=mask)

def bitwise_and(input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Check input and other are integral or boolean tensors
    allowed_dtypes = (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    if input.dtype not in allowed_dtypes:
        raise RuntimeError("input must be an integral or boolean tensor")
    if other.dtype not in allowed_dtypes:
        raise RuntimeError("other must be an integral or boolean tensor")
    
    # Compute broadcasted shape
    try:
        bc_shape = torch.broadcast_shapes(input.shape, other.shape)
    except ValueError as e:
        raise RuntimeError("input and other are not broadcastable") from e
    
    # Determine the result dtype
    dtype = torch.result_type(input, other)
    
    # Broadcast and cast input and other to the result dtype
    input_bc = input.to(dtype).broadcast_to(bc_shape)
    other_bc = other.to(dtype).broadcast_to(bc_shape)
    
    # Ensure inputs are contiguous
    input_contig = input_bc.contiguous()
    other_contig = other_bc.contiguous()
    
    # Prepare output tensor
    if out is None:
        out = torch.empty(bc_shape, dtype=dtype, device=input.device)
        out_contig = out.contiguous()
    else:
        if out.shape != bc_shape:
            raise RuntimeError("out tensor shape does not match broadcasted shape")
        if out.dtype != dtype:
            raise RuntimeError(f"out tensor dtype {out.dtype} does not match expected {dtype}")
        out_contig = out if out.is_contiguous() else torch.empty_like(out, memory_format=torch.contiguous_format)
    
    # Ensure all tensors are on the same device
    if input_contig.device != other_contig.device or out_contig.device != input_contig.device:
        raise RuntimeError("All tensors must be on the same device")
    
    n_elements = out_contig.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    bitwise_and_kernel[grid](
        input_contig, other_contig, out_contig,
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    # Copy back if out was non-contiguous
    if out is not None and not out.is_contiguous():
        out.copy_(out_contig)
    
    return out
