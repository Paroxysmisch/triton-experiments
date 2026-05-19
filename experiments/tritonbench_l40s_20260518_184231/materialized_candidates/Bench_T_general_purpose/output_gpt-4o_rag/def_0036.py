import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh

@triton.jit
def add_gelu_none_kernel(input_ptr, other_ptr, output_ptr, alpha, n_elements):
    pid = tl.program_id(0)
    block_size = 1024
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    result = input + alpha * other
    gelu_result = 0.5 * result * (1 + erf(result * 0.7071067811))
    tl.store(output_ptr + offsets, gelu_result, mask=mask)

@triton.jit
def add_gelu_tanh_kernel(input_ptr, other_ptr, output_ptr, alpha, n_elements):
    pid = tl.program_id(0)
    block_size = 1024
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    result = input + alpha * other
    gelu_result = 0.5 * result * (1 + tanh(0.79788456 * (result + 0.044715 * result * result * result)))
    tl.store(output_ptr + offsets, gelu_result, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    if approximate == 'none':
        add_gelu_none_kernel[grid](input, other, out, alpha, n_elements, BLOCK_SIZE=1024)
    elif approximate == 'tanh':
        add_gelu_tanh_kernel[grid](input, other, out, alpha, n_elements, BLOCK_SIZE=1024)
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")

    return out
