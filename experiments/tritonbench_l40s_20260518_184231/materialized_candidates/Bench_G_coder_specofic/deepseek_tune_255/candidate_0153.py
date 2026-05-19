import torch
import triton
import triton.language as tl

@triton.jit
def sin_kernel(in_ptr0, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Triton kernel to compute the sine of an input array
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.sin(x)
    tl.store(out_ptr + offsets, y, mask=mask)

def sin_triton(x):
    # Function to invoke the Triton kernel
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sin_kernel[grid](x, out, n_elements, BLOCK_SIZE=4)
    return out
