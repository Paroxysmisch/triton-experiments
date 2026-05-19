import torch
import triton
import triton.language as tl
from triton.language.libdevice import asin as libdevice_asin

@triton.jit
def asin(x):
    return libdevice_asin(x)

def test_asin(func_inputs):
    input, = func_inputs
    triton_out = asin(torch.tensor(input))
    torch_out = torch.asin(torch.tensor(input))
    return torch_out == triton_out
