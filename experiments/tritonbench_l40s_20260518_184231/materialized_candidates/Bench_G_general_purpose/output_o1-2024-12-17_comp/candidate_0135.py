import triton
import triton.language as tl
import torch

@triton.jit
def kernel_function(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.math.sin(x)
    tl.store(output_ptr + offsets, y, mask=mask)

def call_kernel(x, BLOCK_SIZE=1024):
    n_elements = x.numel()
    output = torch.empty_like(x)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    kernel_function[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
