import torch
import triton
import triton.language as tl
from torch import Tensor
from .avg_pool1d import avg_pool1d

@triton.jit
def cos_avg_pool1d_triton(
    input, weight, bias, stride, padding, out, x_stride, w_stride, M, N, K, 
    Kc: tl.constexpr, s: tl.constexpr, SB: tl.constexpr, SC: tl.constexpr
):
    # compute offset
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    # calculate position
    rx = pid * SB + tl.arange(0, SB)[:, None]
    ry = bid * SC + tl.arange(0, SC)[None, :]
    # clamp
    rx = tl.minimum(rx, M - 1)
    ry = tl.minimum(ry, N - 1)
    # get mask
    rx_mask = rx < M
    ry_mask = ry < N
    # compute input ptr
    input_ptr = input + bid * x_stride + (rx * K + ry * Kc)
    # compute weight ptr
    weight_ptr = weight + (ry * K + rx * Kc)
    # compute bias ptr
    bias_ptr = bias + ry
    # load
    x = tl.load(input_ptr, mask=rx_mask & ry_mask, other=0.0)
    w = tl.load(weight_ptr, mask=rx_mask & ry_mask, other=0.0)
    b = tl.load(bias_ptr, mask=ry_mask, other=0.0)
    # add
    x = x + b
    # cos
    x = tl.cos(x)
    # mul
    x = x * w
    # reduce
    x = tl.sum(x, axis=1)
    # store
    tl.store(out + bid * x_stride + rx, x, mask=rx_mask)

def cos_avg_pool1d(input: Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> Tensor:
    assert input.dim() == 3, "only accept 3D tensor now"
    # kernel size
    kernel_size = (kernel_size,)
    # stride
    stride = stride if stride is not None else kernel_size
    stride = (stride,)
    # padding
    padding = padding if isinstance(padding, (tuple, list)) else (padding,)
    # device
    device = input.device
    # sequeeze
    pack = False
    if len(kernel_size) == 1:
        kernel_size = kernel_size * 2
        stride = stride * 2
        padding = padding * 2
        pack = True
    # call avg_pool1d
    out = avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    # transpose
    if pack:
        out = out.transpose(1, 2).contiguous()
    return out
