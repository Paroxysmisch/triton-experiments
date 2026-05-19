import torch
import triton
import triton.language as tl
from deepspeed.accelerator import get_accelerator

@triton.jit
def erf_kernel(x):
    # Triton kernel for computing the error function
    return tl.math.erf(x)

def erf(input, *, out=None):
    # Function to compute the error function of a tensor
    if out is None:
        out = torch.empty_like(input, dtype=torch.get_autocast_gpu_dtype() if get_accelerator().on_accelerator() else input.dtype, device=input.device)
    assert input.is_contiguous()
    assert out.is_contiguous()
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    erf_kernel[grid](input, out=out, BLOCK_SIZE=1024)
    return out
