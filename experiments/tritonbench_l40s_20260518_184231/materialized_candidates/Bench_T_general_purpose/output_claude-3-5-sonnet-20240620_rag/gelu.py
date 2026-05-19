import torch
import triton
import triton.language as tl
from triton.language.math import erf, pow, tanh
import math

@triton.jit
def gelu_kernel(
    x_ptr,  # pointer to input tensor
    output_ptr,  # pointer to output tensor
    n_elements,  # number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # size of the block for parallel processing
    approximate: tl.constexpr,  # whether to use tanh approximation
):
    # Calculate the absolute position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Convert to float32 for better precision
    x_f32 = x.to(tl.float32)
    
    if approximate:
        # Tanh approximation
        # GELU(x) = 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x^3)))
        x_cubed = pow(x_f32, 3)
        inner = 0.79788456  # √(2/π)
        inner = inner * (x_f32 + 0.044715 * x_cubed)
        result = 0.5 * x_f32 * (1.0 + tanh(inner))
    else:
        # Exact computation using error function
        # GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x/√2))
        result = 0.5 * x_f32 * (1.0 + erf(x_f32 * 0.7071067811865476))  # 1/√2
        
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

class GELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, approximate='none'):
        # Output tensor
        output = torch.empty_like(x)
        
        # Calculate grid and block sizes
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel
        use_approx = approximate == 'tanh'
        gelu_kernel[(grid,)](
            x_ptr=x,
            output_ptr=output,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            approximate=use_approx,
        )
        
        return output

def gelu(input, approximate='none'):
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function.
    
    Args:
        input (Tensor): Input tensor
        approximate (str): Approximation method ('none' or 'tanh')
    
    Returns:
        Tensor: Output tensor
    """
    if approximate not in ['none', 'tanh']:
        raise ValueError("approximate argument must be either 'none' or 'tanh'")
    
    return GELU.apply(input, approximate)
