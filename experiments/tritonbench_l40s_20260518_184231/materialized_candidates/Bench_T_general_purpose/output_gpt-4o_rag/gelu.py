import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh, pow

@triton.jit
def gelu_none_kernel(x_ptr, output_ptr, n_elements):
    # Calculate the position of each element processed by this instance of the kernel
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load data
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute GELU using the error function approximation
    x_fp32 = x.to(tl.float32)
    gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))

    # Store the result
    tl.store(output_ptr + offsets, gelu, mask=offsets < n_elements)

@triton.jit
def gelu_tanh_kernel(x_ptr, output_ptr, n_elements):
    # Calculate the position of each element processed by this instance of the kernel
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load data
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute GELU using the tanh approximation
    x_fp32 = x.to(tl.float32)
    gelu = 0.5 * x_fp32 * (1 + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32, 3))))

    # Store the result
    tl.store(output_ptr + offsets, gelu, mask=offsets < n_elements)

def gelu(input, approximate='none'):
    # Determine the number of elements in the input tensor
    n_elements = input.numel()
    # Allocate output tensor
    output = torch.empty_like(input)

    # Choose the appropriate kernel based on the 'approximate' argument
    if approximate == 'none':
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        gelu_none_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    elif approximate == 'tanh':
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        gelu_tanh_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")

    return output
