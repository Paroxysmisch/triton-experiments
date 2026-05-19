import triton
import triton.language as tl

@triton.jit
def erfc_kernel(input_ptr, output_erfc_ptr, output_sqrt_ptr, n_elements):
    pid = tl.program_id(0)
    block_size = 1024
    start = pid * block_size
    end = tl.min(start + block_size, n_elements)

    for i in range(start, end):
        x = tl.load(input_ptr + i)
        # Compute erfc using the series expansion
        erfc_val = 1 - (2 / tl.sqrt(tl.pi)) * tl.integrate(tl.exp(-tl.square(tl.arange(0, x))), 0, x)
        sqrt_val = tl.sqrt(x)
        tl.store(output_erfc_ptr + i, erfc_val)
        tl.store(output_sqrt_ptr + i, sqrt_val)

from typing import Tuple
import torch

def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n_elements = input.numel()
    output_erfc = torch.empty_like(input)
    output_sqrt = torch.empty_like(input)

    # Launch the Triton kernel
    grid = (n_elements + 1023) // 1024  # Calculate number of blocks
    erfc_kernel[grid](input, output_erfc, output_sqrt, n_elements)

    return output_erfc, output_sqrt
