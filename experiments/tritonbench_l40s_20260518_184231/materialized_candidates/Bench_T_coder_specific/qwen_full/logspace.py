import torch
import triton
import triton.language as tl

@triton.jit
def _logspace(start, end, steps, base, out, exp_fn, dtype):
    # Initialize index
    i = 0
    # Loop through each step
    while i < steps:
        # Calculate exponent
        exp = start + i * (end - start) / (steps - 1)
        # Calculate value and cast to specified dtype
        out[i] = exp_fn(exp).to(dtype)
        # Increment index
        i += 1

def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    # Wrapper function to call the Triton kernel
    return _logspace_wrapper(
        start, end, steps, base, out, dtype, layout, device, requires_grad
    )
