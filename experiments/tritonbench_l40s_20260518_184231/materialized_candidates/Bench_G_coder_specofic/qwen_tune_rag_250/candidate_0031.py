import torch
import triton
import triton.language as tl

@triton.jit
def mul_kernel(src, dst, BLOCK_SIZE: tl.constexpr):
    exponent_compensator: tl.constexpr = 2.0 ** (127 - 15)
    idxs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + idxs)
    y = x * exponent_compensator
    tl.store(dst + idxs, y)

def launch_mul_kernel(src, BLOCK_SIZE=1):
    dst = torch.empty(src.shape, dtype=torch.float32, device='cuda')
    mul_kernel[(src.shape[0] // BLOCK_SIZE,)](src, dst, BLOCK_SIZE)
    return dst
