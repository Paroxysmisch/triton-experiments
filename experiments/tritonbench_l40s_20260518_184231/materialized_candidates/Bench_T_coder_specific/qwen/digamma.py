import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def digamma_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    indices = indices[indices < n_elements]
    
    x = tl.load(x_ptr + indices)
    y = tl.zeros_like(x)
    
    # Numerical approximation for the derivative of gamma function
    epsilon = 1e-6
    gamma_x = tl.math.gamma(x)
    gamma_x_plus_epsilon = tl.math.gamma(x + epsilon)
    gamma_x_minus_epsilon = tl.math.gamma(x - epsilon)
    
    gamma_prime_x = (gamma_x_plus_epsilon - gamma_x_minus_epsilon) / (2 * epsilon)
    
    y = gamma_prime_x / gamma_x
    
    tl.store(y_ptr + indices, y)

# Define the wrapper function
def digamma(input, out=None):
    assert isinstance(input, torch.Tensor), "Input must be a PyTorch tensor"
    
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    digamma_kernel[(grid_size,), (BLOCK_SIZE,)](input.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

# Constants
BLOCK_SIZE = 256
