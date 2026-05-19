import triton
import triton.language as tl
import torch
from typing import *

# Triton kernel for computing signbit and bitwise AND operations
@triton.jit
def signbit_bitwise_and_kernel(x, y, o_signbit, o_bitwise_and, x_numel, X_BLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * X_BLOCK
    x_index = x_offset + tl.arange(0, X_BLOCK)
    x_mask = x_index < x_numel
    
    x_val = tl.load(x + x_index, x_mask)
    
    o_signbit_val = tl.signbit(x_val)
    tl.store(o_signbit + x_index, o_signbit_val, x_mask)
    
    o_bitwise_and_val = x_val.to(tl.int32) & y
    tl.store(o_bitwise_and + x_index, o_bitwise_and_val, x_mask)

# Wrapper function calling the Triton kernel
def signbit_bitwise_and(input, other) -> Tuple[torch.Tensor, torch.Tensor]:
    if input.dtype == torch.bool:
        other = other.to(torch.int64)

    assert input.is_contiguous(), "Input must be contiguous"
    x_numel = input.numel()
    
    o_signbit = torch.empty_like(input, dtype=torch.bool)
    o_bitwise_and = torch.empty_like(input)
    
    grid = lambda meta: (triton.cdiv(x_numel, meta['X_BLOCK']), )
    signbit_bitwise_and_kernel[grid](input, other, o_signbit, o_bitwise_and, x_numel)
    
    if o_bitwise_and.dtype != input.dtype:
        o_bitwise_and = o_bitwise_and.to(input.dtype)
        
    return o_signbit, o_bitwise_and
