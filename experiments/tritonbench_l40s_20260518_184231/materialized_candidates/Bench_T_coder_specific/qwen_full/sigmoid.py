import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(x):
    # Convert input to float32
    x_fp32 = x.to(tl.float32)
    # Compute sigmoid using the expit function from torch.special
    return torch.special.expit(x_fp32)

def sigmoid(input, *, out=None):
    # Wrapper function for sigmoid operation
    if out == None:
        # Call the Triton kernel if output is not provided
        return sigmoid_kernel(input)
    else:
        # Ensure input and output are on the same device
        assert input.device == out.device
        # Call the Triton kernel with output tensor
        return sigmoid_kernel(input, out=out)
