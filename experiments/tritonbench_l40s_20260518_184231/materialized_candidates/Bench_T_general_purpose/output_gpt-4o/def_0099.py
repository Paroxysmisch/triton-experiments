import triton
import triton.language as tl

@triton.jit
def gelu_kernel(input_ptr, output_ptr, n_elements, approximate, BLOCK_SIZE: tl.constexpr):
    # Create a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Apply GELU
    if approximate == 0:  # 'none'
        # Exact method using CDF
        sqrt_2 = 1.41421356237  # sqrt(2)
        cdf = 0.5 * (1.0 + tl.erf(x / sqrt_2))
        gelu_x = x * cdf
    else:  # 'tanh'
        # Approximate method using tanh
        sqrt_2_over_pi = 0.7978845608  # sqrt(2/pi)
        c = 0.044715
        tanh_arg = sqrt_2_over_pi * (x + c * x**3)
        gelu_x = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
    
    # Store the result
    tl.store(output_ptr + offsets, gelu_x, mask=mask)


import torch

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    # Validate input
    assert isinstance(input, torch.Tensor), "Input must be a torch.Tensor"
    assert approximate in ['none', 'tanh'], "Approximate must be 'none' or 'tanh'"
    
    # Prepare input tensor for Triton
    input_flat = input.flatten()
    n_elements = input_flat.numel()
    
    # Allocate output tensor
    if out is None:
        out = torch.empty_like(input_flat)
    
    # Determine approximate mode
    approximate_mode = 0 if approximate == 'none' else 1
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024  # You can tune this parameter
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_kernel[grid](input_flat, out, n_elements, approximate_mode, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape output to original shape
    gelu_activated = out.view_as(input)
    
    # Compute standard deviation
    mean = gelu_activated.mean(dim=dim, keepdim=True)
    variance = ((gelu_activated - mean) ** 2).sum(dim=dim, keepdim=keepdim) / max(1, gelu_activated.size(dim) - correction)
    std_dev = torch.sqrt(variance)
    
    return std_dev
