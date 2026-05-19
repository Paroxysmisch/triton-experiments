import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, y_ptr, N):
    pid = tl.program_id(0)
    block_start = pid * 1024
    offsets = block_start + tl.arange(0, 1024)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > 0, x, 0)
    tl.store(y_ptr + offsets, y, mask=mask)

def relu(x):
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, 1024),)
    relu_kernel[grid](x, y, N)
    return y
