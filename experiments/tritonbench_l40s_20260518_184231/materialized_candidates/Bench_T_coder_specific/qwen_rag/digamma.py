import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def digamma_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # Get the program ID for the current block
    block_start = pid * BLOCK_SIZE  # Calculate the starting index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Generate offsets for elements in the block
    mask = offsets < n_elements  # Ensure we don't go out of bounds
    x = tl.load(x_ptr + offsets, mask=mask)  # Load input tensor values
    
    # Approximate the digamma function using numerical differentiation
    x_plus_eps = x + eps
    gamma_x = libdevice.tgamma(x)
    gamma_x_plus_eps = libdevice.tgamma(x_plus_eps)
    digamma_approx = (gamma_x_plus_eps - gamma_x) / eps
    
    tl.store(y_ptr + offsets, digamma_approx, mask=mask)  # Store the result in the output tensor
