import triton
import triton.language as tl
from typing import Optional
import torch

@triton.jit
def floor_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the absolute position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute floor
    # For integers, this is a no-op (will just copy)
    # For floating point, computes actual floor
    result = tl.floor(x)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def floor(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    elif out.shape != input.shape:
        raise RuntimeError("out tensor must have same shape as input tensor")
    
    # For integer inputs, just return a copy
    if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, 
                      torch.uint8, torch.bool]:
        return input.clone() if out is None else out.copy_(input)
    
    # Calculate grid and block sizes
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(input.numel(), BLOCK_SIZE),)
    
    # Launch kernel
    floor_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=input.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
