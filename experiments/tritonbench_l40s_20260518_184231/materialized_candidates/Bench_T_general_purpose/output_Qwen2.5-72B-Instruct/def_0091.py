import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def erfc_sqrt_kernel(
    output_erfc: tl.tensor, 
    output_sqrt: tl.tensor, 
    input: tl.tensor, 
    N: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input + offsets, mask=mask)
    
    erfc_val = 1.0 - (2.0 / tl.sqrt(tl.pi)) * tl.erf(x)
    sqrt_val = tl.sqrt(x)
    
    tl.store(output_erfc + offsets, erfc_val, mask=mask)
    tl.store(output_sqrt + offsets, sqrt_val, mask=mask)

def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure the input tensor is on the same device
    device = input.device
    N = input.numel()

    # Allocate output tensors
    output_erfc = torch.empty_like(input)
    output_sqrt = torch.empty_like(input)

    # Define the grid and block sizes
    BLOCK_SIZE = 1024
    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    erfc_sqrt_kernel[grid, BLOCK_SIZE](
        output_erfc, output_sqrt, input, N
    )

    return output_erfc, output_sqrt
