import triton
import triton.language as tl
import torch
import math
from typing import Tuple

# Triton kernel to compute erfc and sqrt of elements in a tensor
@triton.jit
def _erfc_sqrt_kernel(a_ptr, erfc_ptr, sqrt_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create mask for boundary conditions
    mask = offset < n_elements
    # Load input value
    x = tl.load(a_ptr + offset, mask=mask).to(tl.float32)
    
    # Compute sqrt
    sqrt_val = tl.sqrt(x)
    
    # Polynomial approximation of erf(x) (Abramowitz and Stegun 7.1.26)
    # erf(x) ≈ 1 - p(t) * exp(-x^2), with t = 1 / (1 + p*x)
    A1 = 0.254829592
    A2 = -0.284496736
    A3 = 1.421413741
    A4 = -1.453152027
    A5 = 1.061405429
    P  = 0.3275911

    sign = tl.where(x < 0, -1.0, 1.0)
    abs_x = tl.abs(x)
    t = 1.0 / (1.0 + P * abs_x)
    erf_approx = 1.0 - (((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t * tl.exp(-abs_x * abs_x))
    erf_approx = sign * erf_approx

    # erfc(x) = 1 - erf(x)
    erfc_val = 1.0 - erf_approx

    # Store results
    tl.store(sqrt_ptr + offset, sqrt_val, mask=mask)
    tl.store(erfc_ptr + offset, erfc_val, mask=mask)

# Wrapper function to compute both erfc and sqrt
def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out_erfc = torch.empty_like(input)
    out_sqrt = torch.empty_like(input)
    n_elements = input.numel()
    block_size = 128
    grid = triton.cdiv(n_elements, block_size)
    
    _erfc_sqrt_kernel[(grid,)](
        input,       # a_ptr
        out_erfc,    # erfc_ptr
        out_sqrt,    # sqrt_ptr
        n_elements,  # total number of elements
        block_size
    )
    return out_erfc, out_sqrt
