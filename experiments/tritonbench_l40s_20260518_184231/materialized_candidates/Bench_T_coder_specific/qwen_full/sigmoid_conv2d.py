import torch
import triton
import triton.language as tl
from triton.language.libdevice import sigmoid as _sigmoid

@triton.jit
def sigmoid(x):
    """Tanh."""
    return _sigmoid(x)

@triton.jit
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Conv2d."""
    if type(stride) == int:
        stride = (stride, stride)
    if type(padding) == int:
        padding = (padding, padding)
    if type(dilation) == int:
        dilation = (dilation, dilation)

    batch, in_channel, in_h, in_w = input.shape
    out_channel, channel_group, kernel_h, kernel_w = weight.shape
    assert in_channel == channel_group * groups
    assert batch >= 1 and in_channel >= 1 and out_channel >= 1 and kernel_h >= 1 and kernel_w >= 1

    input = tl.pad(input, [(0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])], 0)
    bias = bias if bias is None else bias
    # out = tl.zeros([batch, out_channel, in_h, in_w], tl.float32)
    out = tl.zeros([batch, out_channel, in_h - kernel_h + 2 * padding[0] + 1, in_w - kernel_w + 2 * padding[1] + 1], tl.float32)

    input = input.to(tl.float32)
    weight = weight.to(tl.float32)

    for b in range(batch):
        for g in range(groups):
            cur_input = tl.view(input[b, g * channel_group : (g + 1) * channel_group, :, :], [in_channel, in_h, in_w])
            cur_weight = tl.view(weight[:, g * channel_group : (g + 1) * channel_group, :, :], [channel_group, out_channel, kernel_h, kernel_w])
            for c in range(0, channel_group):
                cur_in = tl.view(cur_input[c, :, :], [in_h, in_w])
                cur_weight_ = tl.view(cur_weight[c, :, :, :], [out_channel, kernel_h, kernel_w])
                for i in range(0, in_h - kernel_h + 2 * padding[0] + 1, stride[0]):
                    for j in range(0, in_w - kernel_w + 2 * padding[1] + 1, stride[1]):
                        window = tl.view(cur_in[i : i + kernel_h, j : j + kernel_w], [kernel_h, kernel_w])
                        out_ = tl.sum(window * cur_weight_, [0, 1])
                        out[b, g * out_channel + c, i + stride[0] * tl.arange(0, tl.num_programs(0)), j + stride[1] * tl.arange(0, tl.num_programs(1))] = out_

    out = out.to(input.dtype)
    return out

@triton.jit
def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Sigmoid conv2d."""
    out = conv2d(input, weight, bias, stride, padding, dilation, groups)
    out = sigmoid(out)
    return out
