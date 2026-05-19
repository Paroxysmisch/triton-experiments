import math
import torch
import triton
import triton.language as tl

@triton.jit
def _logit_kernel(
    in_ptr, out_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Each program handles a contiguous block of elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Check if eps < 0 (which signals None in Python wrapper)
    has_eps = eps >= 0.0
    
    # If eps is available, clamp to [eps, 1-eps]
    # Otherwise, if x < 0 or x > 1 => yield NaN
    z = tl.where(has_eps, tl.where(x < eps, eps, tl.where(x > 1 - eps, 1 - eps, x)), x)
    
    # For eps=None, we must mark invalid => NaN
    valid_mask = tl.where(has_eps, True, (x >= 0) & (x <= 1))
    
    # Compute logit
    # log(z / (1 - z)) = log(z) - log(1 - z)
    num = z
    den = 1.0 - z
    y = tl.log(num) - tl.log(den)
    
    # Apply NaN where invalid for eps=None
    nan_val = 0.0 / 0.0
    y = tl.where(valid_mask, y, nan_val)
    
    # Store
    tl.store(out_ptr + offsets, y, mask=mask)

def logit(input, eps=None, *, out=None):
    """
    logit(input, eps=None, *, out=None) -> Tensor
    
    Returns a new tensor with the logit of the elements of 'input'.
    The input is clamped to [eps, 1 - eps] when eps is not None.
    When eps is None and input < 0 or input > 1, the function yields NaN.

    Parameters:
        input (Tensor): the input tensor.
        eps (float, optional): the epsilon for input clamp bound. Default: None.
        out (Tensor, optional): the output tensor.
    """
    if out is None:
        out = torch.empty_like(input)
    
    # Flatten input and out for 1D kernel launch
    x_flat = input.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)
    
    n_elements = x_flat.numel()
    # Use a power-of-2 block size
    block_size = triton.next_power_of_2(n_elements) if n_elements > 1 else 1
    
    # Map eps=None to eps<0 for kernel
    eps_val = -1.0 if eps is None else float(eps)
    
    grid = ( (n_elements + block_size - 1) // block_size, )
    
    _logit_kernel[grid](
        x_flat, out_flat,
        n_elements,
        eps_val,
        BLOCK_SIZE=block_size
    )
    
    return out.view_as(input)
