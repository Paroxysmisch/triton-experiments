import torch
import triton
import triton.language as tl

@triton.jit
def reciprocal_kernel(x):
    # Convert input to float32 and compute the reciprocal
    x = x.to(tl.float32)
    return 1 / x

def reciprocal(input, *, out=None):
    # Wrapper function for the reciprocal kernel
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape and out.dtype == input.dtype
    # Call the Triton kernel for reciprocal calculation
    reciprocal_kernel[(input,)](out)
    return out
