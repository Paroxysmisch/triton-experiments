import triton
import triton.language as tl
import torch

@triton.jit
def erf_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the block index and position
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to prevent out-of-bounds memory access
    mask = offsets < n_elements
    
    # Load the input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Constants for approximation
    a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    
    # Sign of x
    sign = tl.where(x >= 0, 1.0, -1.0)
    abs_x = tl.abs(x)
    
    # Compute t
    t = 1.0 / (1.0 + p * abs_x)
    
    # Compute approximation of erf
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * tl.exp(-abs_x * abs_x)
    
    # Apply the sign
    erf_x = sign * y
    
    # Store the result
    tl.store(output_ptr + offsets, erf_x, mask=mask)

def erf(input, *, out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("Output must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as the input tensor")
    
    # Get the number of elements
    n_elements = input.numel()
    
    # Define the block size
    BLOCK_SIZE = 1024  # This can be tuned for performance
    
    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    erf_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
