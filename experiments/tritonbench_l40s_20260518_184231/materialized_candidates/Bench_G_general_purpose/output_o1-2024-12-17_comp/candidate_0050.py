import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x = tl.where(x < 0, 0, x)
    tl.store(out_ptr + offsets, x, mask=mask)

def relu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: ( (N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    relu_kernel[grid](x, out, N, BLOCK_SIZE=1024)
    return out
