import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh

@triton.jit
def fused_masked_select_add_gelu_kernel(
    input_ptr, mask_ptr, other_ptr, output_ptr, alpha, N, approximate
):
    pid = tl.program_id(0)
    # Create a pointer for each element in the input
    offsets = pid * N + tl.arange(0, N)
    
    # Load data from input, mask, and other tensors
    x = tl.load(input_ptr + offsets)
    m = tl.load(mask_ptr + offsets, mask=offsets < N, other=0)
    o = tl.load(other_ptr + offsets, mask=offsets < N, other=0)
    
    # Perform masked selection
    z = tl.where(m, x, 0)
    
    # Perform addition with scaling
    s = z + alpha * o
    
    # Apply GELU activation
    if approximate == 0:  # 'none'
        y = 0.5 * s * (1 + erf(s * 0.7071067811))
    else:  # 'tanh'
        y = 0.5 * s * (1 + tanh(s * 0.79788456 * (1 + 0.044715 * s * s)))
    
    # Store the result
    tl.store(output_ptr + offsets, y, mask=offsets < N)

def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    # Ensure inputs are on the same device
    assert input.device == mask.device == other.device, "All inputs must be on the same device"
    
    # Determine the size of the input
    N = input.numel()
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Map 'approximate' parameter to a numerical value
    approximate_value = 0 if approximate == 'none' else 1
    
    # Launch the Triton kernel
    grid = (N // 1024 + 1,)
    fused_masked_select_add_gelu_kernel[grid](
        input_ptr=input, mask_ptr=mask, other_ptr=other, output_ptr=out, 
        alpha=alpha, N=N, approximate=approximate_value
    )
    
    return out
