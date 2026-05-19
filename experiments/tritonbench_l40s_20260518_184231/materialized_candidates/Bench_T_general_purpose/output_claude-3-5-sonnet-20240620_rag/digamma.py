import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Digamma Kernel
# This kernel computes the digamma function for each element in the input tensor.
@triton.jit
def digamma_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # Get the program ID for the current block
    block_start = pid * BLOCK_SIZE  # Calculate the starting index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Generate offsets for elements in the block
    mask = offsets < n_elements  # Ensure we don't go out of bounds
    x = tl.load(input_ptr + offsets, mask=mask)  # Load input tensor values

    # Compute the digamma function using the relation: digamma(x) = Γ'(x) / Γ(x)
    # Here we use libdevice functions for gamma and its derivative
    gamma_x = libdevice.gamma(x)  # Compute Γ(x)
    gamma_prime_x = libdevice.gamma_prime(x)  # Compute Γ'(x)
    result = gamma_prime_x / gamma_x  # Compute digamma

    # Handle the case for input 0, returning -Inf
    result = tl.where(x == 0, -tl.inf, result)  # Set result to -Inf for input 0

    tl.store(output_ptr + offsets, result, mask=mask)  # Store the result in the output tensor

# Wrapper function for digamma
def digamma(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)  # Create an output tensor if not provided

    n_elements = input.numel()  # Get the number of elements in the tensor
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )  # Define the grid dimensions for Triton

    digamma_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)  # Launch the kernel
    return out  # Return the output tensor
