import triton
import triton.language as tl

@triton.jit
def chebyshev_kernel(input_ptr, n_ptr, output_ptr, input_n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_n

    input = tl.load(input_ptr + offsets, mask=mask)
    n = tl.load(n_ptr + offsets, mask=mask)

    # Initialize T0 and T1
    T0 = tl.where(n == 0, 1.0, 0.0)
    T1 = tl.where(n == 1, input, 0.0)

    # Compute Tn using the recursive formula or trigonometric formula
    Tn = tl.where(n == 0, T0, T1)
    Tn_minus_1 = T0
    Tn_minus_2 = T0

    for i in range(2, 6):
        Tn = tl.where(n == i, 2 * input * Tn_minus_1 - Tn_minus_2, Tn)
        Tn_minus_2 = Tn_minus_1
        Tn_minus_1 = Tn

    # For n >= 6 and |input| <= 1, use the trigonometric formula
    Tn = tl.where((n >= 6) & (tl.abs(input) <= 1.0), tl.cos(n * tl.acos(input)), Tn)

    # Store the result
    tl.store(output_ptr + offsets, Tn, mask=mask)

import torch

def chebyshev_polynomial_t(input, n, *, out=None):
    # Ensure input and n are tensors
    input = input.contiguous()
    n = n.contiguous()

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        out = out.contiguous()

    # Define the grid and block sizes
    BLOCK_SIZE = 256
    grid = (input.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    chebyshev_kernel[grid, BLOCK_SIZE](
        input, n, out, input.numel(), BLOCK_SIZE=BLOCK_SIZE
    )

    return out
