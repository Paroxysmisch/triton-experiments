import math
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def cos_signbit_func(a_ptr, cos_ptr, sign_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load input
    a_val = tl.load(a_ptr + offset, mask=mask)

    # Compute cosine
    b_val = tl.cos(a_val.to(tl.float32))

    # Determine sign bit (negative = 1, non-negative = 0)
    sign_val = b_val < 0
    sign_val = tl.where(sign_val, 1, 0)

    # Store results
    tl.store(cos_ptr + offset, b_val, mask=mask)
    tl.store(sign_ptr + offset, sign_val, mask=mask)

def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n_elements = input.numel()
    # Output for cosine
    cos_out = torch.empty_like(input, dtype=torch.float32)
    # Intermediate integer output for sign bit
    sign_int = torch.empty_like(input, dtype=torch.int32)

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)

    cos_signbit_func[(grid_size, 1, 1)](
        input, 
        cos_out, 
        sign_int, 
        n_elements, 
        block_size
    )

    # Convert integer sign storage to boolean
    sign_bool = sign_int.bool()

    return cos_out, sign_bool
