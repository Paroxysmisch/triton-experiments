import triton
import triton.language as tl
import torch
from collections import namedtuple

# Triton kernel for GELU and minimum computation
@triton.jit
def gelu_min_kernel(input_ptr, output_ptr, index_ptr, n_elements, method, dim_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Apply GELU
    if method == 0:  # 'none'
        cdf = 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        gelu_x = x * cdf
    else:  # 'tanh'
        tanh_part = tl.tanh(tl.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * x * x * x))
        gelu_x = 0.5 * x * (1.0 + tanh_part)
    
    # Initialize min value and index
    min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
    min_idx = tl.full([BLOCK_SIZE], -1, dtype=tl.int32)
    
    # Compute min
    for i in range(dim_size):
        val = gelu_x[i]
        if val < min_val[i]:
            min_val[i] = val
            min_idx[i] = i
    
    # Store results
    tl.store(output_ptr + offsets, min_val, mask=mask)
    tl.store(index_ptr + offsets, min_idx, mask=mask)

# Wrapper function
def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    method = 0 if approximate == 'none' else 1
    input_flat = input if dim is None else input.flatten(start_dim=dim)
    n_elements = input_flat.numel()
    
    # Prepare output tensors
    if out is None:
        out = torch.empty_like(input_flat)
    index_out = torch.empty_like(input_flat, dtype=torch.int32)
    
    # Launch Triton kernel
    grid = (triton.cdiv(n_elements, 1024),)
    gelu_min_kernel[grid](input_flat, out, index_out, n_elements, method, input_flat.size(dim), BLOCK_SIZE=1024)
    
    # Reshape and process output
    if dim is not None:
        if keepdim:
            out = out.view(*input.shape[:dim], 1, *input.shape[dim+1:])
            index_out = index_out.view(*input.shape[:dim], 1, *input.shape[dim+1:])
        else:
            out = out.view(*input.shape[:dim], *input.shape[dim+1:])
            index_out = index_out.view(*input.shape[:dim], *input.shape[dim+1:])
        return namedtuple('GeluMin', ['values', 'indices'])(out, index_out)
    else:
        return out.min()

# Example usage
x = torch.randn(2, 3, 4, device='cuda')
result = gelu_min(x, approximate='tanh', dim=1, keepdim=True)
print(result)
