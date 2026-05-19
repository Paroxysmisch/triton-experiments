import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.libdevice import div_rn


@triton.jit
def leaky_relu(x, negative_slope):
    # Convert input to float32
    x_fp32 = x.to(tl.float32)
    # Compute LeakyReLU
    y = tl.where(x_fp32 >= 0, x_fp32, negative_slope * x_fp32)
    return y


def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    # Check constraints
    assert negative_slope >= 0, "LeakyReLU: negative_slope should be >= 0"
    assert input.is_contiguous(), "LeakyReLU: input tensor must be contiguous"
    # Call Triton kernel
    output = input if inplace else input.clone()
    inp_shape = input.shape
    input = input.reshape(-1)
    output = output.reshape(-1)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    leaky_relu[grid](input, output, negative_slope, n_elements)
    return output.reshape(inp_shape)
