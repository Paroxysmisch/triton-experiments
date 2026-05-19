import torch
import triton
import triton.language as tl

@triton.jit
def airy_ai_kernel(x, *, out):
    # Convert input to float32
    x_fp32 = x.to(tl.float32)
    # Compute Airy function Ai
    y_fp32 = tl.airy_ai(x_fp32)
    # Write result to output
    tl.store(out, y_fp32)

def airy_ai(input, *, out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    # Ensure output is a tensor
    if out is not None and not isinstance(out, torch.Tensor):
        out = torch.tensor(out)
    # Call Triton kernel
    airy_ai_kernel(input, out=out)
    return out
