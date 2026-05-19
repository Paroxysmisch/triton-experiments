import torch
import triton
import triton.language as tl

@triton.jit
def _tile_exp(x):
    # Convert input to float32
    x_fp32 = x.to(tl.float32)
    # Tile the tensor
    y = tl.tile(x_fp32, (2, 3))
    # Apply the exponential function
    z = tl.exp(y)
    return z

def tile_exp(x):
    # Call the Triton kernel
    return _tile_exp(x)
