import triton
import triton.language as tl

@triton.jit
def i0_kernel(input_ptr, output_ptr, n_elements):
    # Compute the zeroth order modified Bessel function of the first kind
    idx = tl.program_id(0)
    if idx >= n_elements:
        return

    x = tl.load(input_ptr + idx)
    sum_result = 0.0
    k = 0

    # Calculate the series expansion
    while True:
        term = (x**2 / 4)**k / (tl.math.factorial(k) ** 2)
        sum_result += term
        if term < 1e-10:  # Break if the term is small enough
            break
        k += 1

    tl.store(output_ptr + idx, sum_result)

import torch

def i0(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure input is a 1D tensor
    if input.dim() != 1:
        raise ValueError("Input tensor must be 1D.")
    
    n_elements = input.numel()
    if out is None:
        out = torch.empty_like(input)

    # Launch the Triton kernel
    grid = (n_elements,)
    i0_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements)

    return out
