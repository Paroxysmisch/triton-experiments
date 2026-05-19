import torch
import triton
import triton.language as tl
from triton.language.math import erf, pow, tanh
from collections import namedtuple
import math

@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    approximate: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Convert to float32 for better precision
    x_f32 = x.to(tl.float32)
    
    # Compute GELU based on approximation method
    if approximate == 0:  # 'none'
        # x * Φ(x)
        result = x_f32 * 0.5 * (1 + erf(x_f32 * 0.7071067811865476))  # 1/sqrt(2)
    else:  # 'tanh'
        # 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        cubic = x_f32 * x_f32 * x_f32
        inner = 0.79788456 * (x_f32 + 0.044715 * cubic)  # sqrt(2/π) ≈ 0.79788456
        result = 0.5 * x_f32 * (1 + tanh(inner))
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def min_reduce_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    n_rows,
    n_cols,
    row_stride,
    col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Initialize min value and index
    min_val = float('inf')
    min_idx = 0
    
    # Compute minimums for this row/segment
    for i in range(0, n_cols, BLOCK_SIZE):
        # Create block mask
        block_mask = tl.arange(0, BLOCK_SIZE) + i < n_cols
        
        # Load values
        offset = pid * row_stride + i * col_stride
        x = tl.load(input_ptr + offset, mask=block_mask, other=float('inf'))
        
        # Update minimum and index
        curr_min = tl.min(x, axis=0)
        if curr_min < min_val:
            min_val = curr_min
            min_idx = i + tl.argmin(x, axis=0)
    
    # Store results
    tl.store(output_ptr + pid, min_val)
    tl.store(indices_ptr + pid, min_idx)

def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    # Input validation
    if approximate not in ['none', 'tanh']:
        raise ValueError("approximate must be either 'none' or 'tanh'")
    
    # Apply GELU first
    gelu_output = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    gelu_kernel[grid](
        input.data_ptr(),
        gelu_output.data_ptr(),
        n_elements,
        BLOCK_SIZE=1024,
        approximate=1 if approximate == 'tanh' else 0,
    )
    
    # If dim is None, find global minimum
    if dim is None:
        result = torch.min(gelu_output)
        return result if out is None else out.copy_(result)
    
    # Otherwise, find minimum along specified dimension
    dim = dim if dim >= 0 else input.dim() + dim
    n_rows = input.size(dim)
    n_cols = input.numel() // n_rows
    
    # Prepare output tensors
    output_shape = list(input.shape)
    if not keepdim:
        output_shape.pop(dim)
    else:
        output_shape[dim] = 1
        
    values = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    indices = torch.empty(output_shape, dtype=torch.long, device=input.device)
    
    # Calculate strides
    row_stride = input.stride(dim)
    col_stride = 1 if dim == input.dim() - 1 else input.stride(dim + 1)
    
    # Launch reduction kernel
    grid = lambda meta: (n_cols,)
    min_reduce_kernel[grid](
        gelu_output.data_ptr(),
        values.data_ptr(),
        indices.data_ptr(),
        n_rows,
        n_cols,
        row_stride,
        col_stride,
        BLOCK_SIZE=1024,
    )
    
    # Return results
    result = namedtuple('Result', ['values', 'indices'])(values, indices)
    return result if out is None else out.copy_(result.values)
