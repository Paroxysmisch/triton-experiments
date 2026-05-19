import triton
import triton.language as tl

@triton.jit
def gelu_kernel(input_ptr, output_ptr, n_elements, approximate: tl.constexpr):
    pid = tl.program_id(0)
    BLOCK_SIZE = 1024
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    
    if approximate == 'none':
        # Exact GELU
        phi = 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        gelu_x = x * phi
    elif approximate == 'tanh':
        # Approximate GELU using tanh
        cdf = 0.5 * (1.0 + tl.tanh(tl.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * x * x * x)))
        gelu_x = x * cdf
    
    tl.store(output_ptr + offsets, gelu_x, mask=mask)

import torch

def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    # Determine the number of elements
    n_elements = input.numel()
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](input, out, n_elements, approximate=approximate)
    
    # Perform reduction
    if dim is None:
        min_val = out.min()
    else:
        min_val, _ = out.min(dim=dim, keepdim=keepdim)
    
    return min_val

# Example usage
input_tensor = torch.randn(1024)
result = min_gelu(input_tensor, approximate='tanh')
