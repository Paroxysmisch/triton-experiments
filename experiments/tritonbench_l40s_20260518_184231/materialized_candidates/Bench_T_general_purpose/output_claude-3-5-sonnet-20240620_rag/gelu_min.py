import torch
import triton
import triton.language as tl
from triton.language.math import erf, pow, tanh
from collections import namedtuple
import math

@triton.jit
def gelu_none_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    x = x.to(tl.float32)
    
    # GELU exact computation: x * Φ(x)
    sqrt2_inv = 0.7071067811865476  # 1/√2
    gelu = 0.5 * x * (1 + erf(x * sqrt2_inv))
    
    # Store result
    tl.store(output_ptr + offsets, gelu, mask=mask)

@triton.jit
def gelu_tanh_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    x = x.to(tl.float32)
    
    # GELU tanh approximation
    sqrt_2_pi = 0.7978845608028654  # √(2/π)
    gelu = 0.5 * x * (1 + tanh(sqrt_2_pi * (x + 0.044715 * pow(x, 3))))
    
    # Store result
    tl.store(output_ptr + offsets, gelu, mask=mask)

@triton.jit
def reduce_min_kernel(
    x_ptr, values_ptr, indices_ptr,
    stride, n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Initialize min value and index
    min_val = float('inf')
    min_idx = 0
    
    # Compute min along the specified dimension
    row_idx = pid
    if row_idx < n_rows:
        for col_idx in range(0, n_cols):
            offset = row_idx * stride + col_idx
            val = tl.load(x_ptr + offset)
            if val < min_val:
                min_val = val
                min_idx = col_idx
    
    # Store results
    if row_idx < n_rows:
        tl.store(values_ptr + row_idx, min_val)
        tl.store(indices_ptr + row_idx, min_idx)

class GELUMin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, approximate='none', dim=None, keepdim=False):
        # First apply GELU
        output = torch.empty_like(input)
        n_elements = input.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        if approximate == 'none':
            gelu_none_kernel[grid](
                input, output,
                n_elements,
                BLOCK_SIZE=1024
            )
        elif approximate == 'tanh':
            gelu_tanh_kernel[grid](
                input, output,
                n_elements,
                BLOCK_SIZE=1024
            )
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")
        
        # Then compute min
        if dim is None:
            return output.min()
        else:
            values = torch.empty(input.shape[:dim] + input.shape[dim+1:], 
                               device=input.device, dtype=input.dtype)
            indices = torch.empty_like(values, dtype=torch.long)
            
            n_rows = values.numel()
            n_cols = input.shape[dim]
            stride = input.stride(dim)
            
            reduce_min_kernel[(n_rows,)](
                output, values, indices,
                stride, n_rows, n_cols,
                BLOCK_SIZE=1024
            )
            
            if keepdim:
                values = values.unsqueeze(dim)
                indices = indices.unsqueeze(dim)
            
            return namedtuple('MinResult', ['values', 'indices'])(values, indices)

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    """
    Applies GELU activation followed by minimum reduction.
    
    Args:
        input (Tensor): Input tensor
        approximate (str): Method for computing GELU ('none' or 'tanh')
        dim (int, optional): Dimension to reduce. If None, returns min of all elements
        keepdim (bool): Whether to keep the reduced dimension
        out (Tensor, optional): Output tensor
    
    Returns:
        Tensor or (Tensor, LongTensor): Minimum values and corresponding indices if dim specified,
                                      otherwise returns minimum value tensor
    """
    return GELUMin.apply(input, approximate, dim, keepdim)
