import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh

@triton.jit
def gelu_none_kernel(input_ptr, other_ptr, alpha, output_ptr, n_elements):
    # Get program ID and calculate range of elements this program will process
    pid = tl.program_id(axis=0)
    offsets = pid * 256 + tl.arange(0, 256)
    mask = offsets < n_elements

    # Load inputs
    input_val = tl.load(input_ptr + offsets, mask=mask)
    other_val = tl.load(other_ptr + offsets, mask=mask) if other_ptr else 0

    # Perform the subtraction and scaling
    x = input_val - alpha * other_val

    # Compute GELU using the error function approximation
    gelu_result = 0.5 * x * (1 + erf(x * 0.7071067811))

    # Store the result
    tl.store(output_ptr + offsets, gelu_result, mask=mask)

@triton.jit
def gelu_tanh_kernel(input_ptr, other_ptr, alpha, output_ptr, n_elements):
    # Get program ID and calculate range of elements this program will process
    pid = tl.program_id(axis=0)
    offsets = pid * 256 + tl.arange(0, 256)
    mask = offsets < n_elements

    # Load inputs
    input_val = tl.load(input_ptr + offsets, mask=mask)
    other_val = tl.load(other_ptr + offsets, mask=mask) if other_ptr else 0

    # Perform the subtraction and scaling
    x = input_val - alpha * other_val

    # Compute GELU using the tanh approximation
    gelu_result = 0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x * x * x)))

    # Store the result
    tl.store(output_ptr + offsets, gelu_result, mask=mask)

def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a Tensor")
    
    # Convert other to tensor if it's a number
    if isinstance(other, (int, float)):
        other = torch.full_like(input, other)
    elif not isinstance(other, torch.Tensor):
        raise TypeError("other must be a Tensor or a Number")
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Get the number of elements
    n_elements = input.numel()
    
    # Select the appropriate kernel
    if approximate == 'none':
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        gelu_none_kernel[grid](input, other, alpha, out, n_elements, BLOCK_SIZE=256)
    elif approximate == 'tanh':
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        gelu_tanh_kernel[grid](input, other, alpha, out, n_elements, BLOCK_SIZE=256)
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")

    return out
