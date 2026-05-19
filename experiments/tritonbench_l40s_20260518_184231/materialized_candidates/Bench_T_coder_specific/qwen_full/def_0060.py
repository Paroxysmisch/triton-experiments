import torch
import triton
import triton.language as tl

@triton.jit
def exp_sqrt_kernel(input, out):
    # Convert input to float32
    input_fp32 = input.to(tl.float32)
    # Compute exponential and then square root
    output = tl.sqrt(tl.exp(input_fp32))
    # Write result to output
    tl.store(output, input)

def exp_sqrt(input, out=None) -> torch.Tensor:
    # Ensure input is a tensor
    assert isinstance(input, torch.Tensor)
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    else:
        assert input.is_cuda == out.is_cuda
    # Call Triton kernel
    exp_sqrt_kernel[(1,)](input, out)
    return out
