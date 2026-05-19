import torch
import triton
import triton.language as tl

@triton.jit
def rand(*size, dtype=tl.float32, device='cuda'):
    return tl.rand(*size, dtype=dtype, device=device)

def rand(*size, dtype=None, layout=None, device=None, pin_memory=None):
    return rand(*size, dtype=dtype, device=device)
