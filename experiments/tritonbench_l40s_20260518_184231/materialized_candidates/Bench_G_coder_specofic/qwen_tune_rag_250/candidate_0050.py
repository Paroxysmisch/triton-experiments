import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(in_ptr0, out_ptr0, N, **meta):
    pid = tl.program_id(0)
    block_start = pid * 1024
    offsets = block_start + tl.arange(0, 1024)
    mask = offsets < N
    x = tl.load(in_ptr0 + offsets, mask=mask)
    out = tl.where(x >= 0, x, 0)
    tl.store(out_ptr0 + offsets, out, mask=mask)

def relu(in_ptr0):
    out_ptr0 = torch.empty_like(in_ptr0)
    N = in_ptr0.numel()
    grid = lambda meta: (triton.cdiv(N, 1024),)
    relu_kernel[grid](in_ptr0, out_ptr0, N, XBLOCK=1024)
    return out_ptr0
