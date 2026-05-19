import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Union

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def gelu_approximate(x):
    # From https://arxiv.org/pdf/1606.08415.pdf
    return 0.5 * x * (1 + tanh(_sqrt(2 / _pi) * (x + 0.044715 * x * x * x)))

@triton.jit
def gelu(x, approximate):
    if approximate == "tanh":
        return gelu_approximate(x)
    else:
        return x * tl.sigmoid(x * 1.702)

@triton.jit
def conv2d(input, weight, bias, stride, padding, dilation, groups, out):
    # Triton's conv2d doesn't support dilation, so we need to handle it manually
    input = _padded_conv2d_input(input, padding, dilation)
    return tl.conv2d(input, weight, bias, stride, 1, groups, out_dtype=out.dtype)

def gelu_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int], str] = 0, dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, approximate: str = 'none', out: Optional[Tensor] = None) -> Tensor:
    if dilation != 1:
        raise NotImplementedError("Dilation > 1 is not supported by this layer")
    if out is None:
        out = torch.empty((input.shape[0], weight.shape[0], input.shape[2] // stride, input.shape[3] // stride), device=input.device, dtype=input.dtype)
    conv2d(input, weight, bias, stride, padding, dilation, groups, out)
    tl.static_print(out)
    return gelu(out, approximate)
