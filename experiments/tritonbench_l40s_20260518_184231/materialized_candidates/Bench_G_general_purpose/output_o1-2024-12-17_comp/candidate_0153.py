import triton
import triton.language as tl
import torch

@triton.jit
def sin_kernel(in_ptr0, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask, other=0.0)
    y = tl.sin(x)
    tl.store(out_ptr + offsets, y, mask=mask)

def sin_triton(x):
    n_elements = x.numel()
    y = torch.empty_like(x)
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    sin_kernel[grid](x, y, n_elements, BLOCK_SIZE=4)
    return y
