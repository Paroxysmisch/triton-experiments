import torch
import triton
import triton.language as tl

@triton.jit
def i0_kernel(x, *, out=None):
    # Convert input to float32
    x_fp32 = x.to(tl.float32)

    # Compute the zeroth order modified Bessel function of the first kind
    res = tl.math.i0(x_fp32)

    # Write result to output
    if out is not None:
        out.copy_to(res, overwrite=True)
    return res

def i0(input, *, out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)

    # Call the Triton kernel
    return i0_kernel(input, out=out)
