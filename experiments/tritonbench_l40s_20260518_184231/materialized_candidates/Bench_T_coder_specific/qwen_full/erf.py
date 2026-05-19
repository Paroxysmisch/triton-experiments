import torch
import triton
import triton.language as tl

@triton.jit
def erf_kernel(input, out):
    # Convert input to float32
    input_fp32 = input.to(tl.float32)
    # Compute the error function
    output = tl.erf(input_fp32)
    # Write the result to the output tensor
    tl.store(out, output)

def erf(input, *, out=None):
    # Determine if input is a number or a tensor
    if isinstance(input, (float, int)):
        input = torch.tensor(input)
    # Ensure input is a tensor
    assert isinstance(input, torch.Tensor)
    # Create an output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    # Call the Triton kernel for error function computation
    erf_kernel[(1,)](input, out)
    return out
