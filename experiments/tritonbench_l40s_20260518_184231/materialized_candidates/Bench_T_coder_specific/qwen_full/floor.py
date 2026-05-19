import torch
import triton
import triton.language as tl

@triton.jit
def floor_kernel(x):
    # Convert input to float32
    x_fp32 = x.to(tl.float32)
    # Calculate floor value
    x_floor = tl.math.floor(x_fp32)
    # Convert result back to original input type
    return x_floor.to(x.dtype)

def floor(input, *, out=None):
    # Ensure input is a tensor
    assert isinstance(input, torch.Tensor), "Input must be a tensor"
    # Call Triton kernel for floor operation
    return floor_kernel(input)
