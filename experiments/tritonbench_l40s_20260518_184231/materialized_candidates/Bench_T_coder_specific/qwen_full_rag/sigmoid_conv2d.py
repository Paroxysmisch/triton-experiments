import torch
import triton
import triton.language as tl
from ..utils import get_triton_grid
from .act_func_conv2d import act_func_conv2d

@triton.jit
def sigmoid(x):
    return 1 / (1 + tl.exp(-x))

@triton.jit
def _sigmoid_conv2d_forward(
    input_pointer, weight_pointer, bias_pointer, output_pointer,
    N, C, H, W, K, Y, X, stride_n_input, stride_c_input, stride_h_input, stride_w_input,
    stride_out_n, stride_out_c, stride_out_y, stride_out_x, 
    padding, stride_kn_weight, stride_kc_weight, stride_ky_weight, stride_kx_weight, 
    group, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, 
    SPLIT_N: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // (Y * X)
    residual = pid % (Y * X)
    y = residual // X
    x = residual % X

    off_group = tl.arange(0, BLOCK_C) // group
    group_id = tl.where(off_group < C, off_group, -1)
    
    off_cn = tl.arange(0, BLOCK_C) % group
    c = group_id * group + off_cn
    
    kn = tl.arange(0, SPLIT_K)
    kw = tl.arange(0, BLOCK_H)
    kh = tl.arange(0, BLOCK_W)

    input_offset = n * stride_n_input + c[:, None] * stride_c_input + (y+kh[:, None]-padding)[:, None] * stride_h_input + (x+kw[None,:]-padding)[None,:] * stride_w_input
    input_mask = (c[:, None] < C) & ((y+kh[:, None]) >= padding) & ((y+kh[:, None]) < H+padding) & ((x+kw[None,:]) >= padding) & ((x+kw[None,:]) < W+padding)
    input = tl.load(input_pointer + input_offset, mask=input_mask, other=0.)
    
    weight_offset = c[:, None, None] * stride_kn_weight + kn[None,:,None] * stride_kc_weight + kh[:,None,None] * stride_ky_weight + kw[None,:,:] * stride_kx_weight
    weight_mask = (kn[:, None, None] < K) & (c[:, None, None] < C)
    weight = tl.load(weight_pointer + weight_offset, mask=weight_mask, other=0.)

    bias_offset = c * stride_out_c
    bias_mask = c < C
    bias = tl.load(bias_pointer + bias_offset, mask=bias_mask, other=0.)

    acc = tl.sum(input * weight, axis=(1, 2, 3)) + bias

    output_offset = n * stride_out_n + c * stride_out_c + y * stride_out_y + x * stride_out_x
    output_mask = (c < C) & ((y+kh[None, :, None]) >= padding) & ((y+kh[None, :, None]) < H+padding) & ((x+kw[:, None, None]) >= padding) & ((x+kw[:, None, None]) < W+padding)
    tl.store(output_pointer + output_offset, sigmoid(acc), mask=output_mask)

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    check_device(input, weight)
    check_device(bias, weight)
    check_contiguous(input, weight)
    check_contiguous(bias, weight)

    if len(input.shape) != 4:
        raise ValueError(f"only accepts 4D input, but got {input.shape}")

    if bias is not None and len(bias.shape) != 1:
        raise ValueError(f"only accepts 1D bias, but got {bias.shape}")

    N, C, H, W = input.shape
    K, _, kH, kW = weight.shape

    if isinstance(stride, int):
        stride = [stride, stride]
    elif len(stride) != 2:
        raise ValueError("stride must either be an integer or a list of two integers")

    if isinstance(dilation, int):
        dilation = [dilation, dilation]
    elif len(dilation) != 2:
        raise ValueError("dilation must either be an integer or a list of two integers")

    if len(padding) == 1:
        padding = [padding[0], padding[0]]
    elif len(padding) == 2:
        padding = [padding[0], padding[1]]
    elif len(padding) == 4:
        padding = [padding[0] + padding[2], padding[1] + padding[3]]
    else:
        raise ValueError("Invalid padding")

    if K % groups != 0:
        raise ValueError('in_channels must be divisible by groups')

    if C % groups != 0:
        raise ValueError('in_channels must be divisible by groups')

    if out is None:
        out = torch.empty(N, C, H, W, device=input.device, dtype=torch.float32)
    else:
        assert out.shape == (N, C, H, W)
        assert out.device == input.device

    def grid(meta):
        return get_triton_grid(N)(meta['SPLIT_N'], meta['BLOCK_C'] // groups, meta['BLOCK_H'], meta['BLOCK_W'])

    act_func_conv2d(
        input, weight, bias, out,
        0, 0, stride[0], stride[1],
        padding[0], padding[1], 0, 0,
        grid
    )

    return out
