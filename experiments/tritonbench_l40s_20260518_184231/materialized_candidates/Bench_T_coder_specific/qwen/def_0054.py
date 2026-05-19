import triton
from typing import Optional, Union, Tuple
import torch

# Define Triton kernel grid size and block size
GRID_SIZE = lambda n: (n + 255) // 256
BLOCK_SIZE = 256

def gelu_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int], str] = 0, dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, approximate: str = 'none', out: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert input.device.type == 'cuda'
    assert weight.device.type == 'cuda'

    I_N, I_C, I_H, I_W = input.shape
    W_O, W_I, W_KH, W_KW = weight.shape

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    oH = (I_H + 2 * padding[0] - dilation[0] * (W_KH - 1) - 1) // stride[0] + 1
    oW = (I_W + 2 * padding[1] - dilation[1] * (W_KW - 1) - 1) // stride[1] + 1

    if out is None:
        out = torch.empty((I_N, W_O, oH, oW), device=input.device, dtype=input.dtype)

    # Prepare inputs for Triton kernel
    I_ptr = input.data_ptr()
    W_ptr = weight.data_ptr()
    B_ptr = bias.data_ptr() if bias is not None else 0
    O_ptr = out.data_ptr()

    # Launch convolution kernel
    grid_size = GRID_SIZE(I_N * W_O * oH * oW)
    conv2d_kernel[I_N * W_O * oH * oW](I_ptr, W_ptr, B_ptr, O_ptr, I_N, I_C, I_H, I_W, W_O, W_I, W_KH, W_KW, stride[0], padding[0], dilation[0], groups, BLOCK_SIZE=BLOCK_SIZE)

    # Launch GELU kernel
    grid_size = GRID_SIZE(out.numel())
    gelu_kernel[out.numel(), BLOCK_SIZE](O_ptr, O_ptr, out.numel(), approximate)

    return out
