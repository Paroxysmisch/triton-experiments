import torch
import triton
import triton.language as tl

@triton.jit
def relu_max_pool2d_conv2d_triton(
    input, weight, bias, stride, padding, dilation, groups, pool_size, pool_stride, pool_padding, pool_dilation, pool_ceil_mode, inplace
):
    input_stride_0 = input.stride(0)
    input_stride_1 = input.stride(1)
    input_stride_2 = input.stride(2)
    input_stride_3 = input.stride(3)
    weight_stride_0 = weight.stride(0)
    weight_stride_1 = weight.stride(1)
    weight_stride_2 = weight.stride(2)
    weight_stride_3 = weight.stride(3)
    if bias is not None:
        bias_stride_0 = bias.stride(0)
    else:
        bias_stride_0 = 0
    stride_0 = stride[0]
    stride_1 = stride[1]
    padding_0 = padding[0]
    padding_1 = padding[1]
    dilation_0 = dilation[0]
    dilation_1 = dilation[1]
    groups_0 = groups
    in_channels = weight.shape[1]
    in_channels_1 = in_channels * dilation_0 * dilation_1
    out_channels = weight.shape[0]
    pool_size_0 = pool_size[0]
    pool_size_1 = pool_size[1]
    if pool_stride is None:
        pool_stride_0 = pool_size_0
        pool_stride_1 = pool_size_1
    else:
        pool_stride_0 = pool_stride[0]
        pool_stride_1 = pool_stride[1]
    pool_padding_0 = pool_padding[0]
    pool_padding_1 = pool_padding[1]
    pool_dilation_0 = pool_dilation[0]
    pool_dilation_1 = pool_dilation[1]
    iH = input.shape[2]
    iW = input.shape[3]
    oH = triton.cdiv(iH + padding_0 * 2 - dilation_0 * (pool_size_0 - 1) - pool_padding_0 * 2, pool_stride_0)
    oW = triton.cdiv(iW + padding_1 * 2 - dilation_1 * (pool_size_1 - 1) - pool_padding_1 * 2, pool_stride_1)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    offset_n = pid_n
    offset_c = (pid_c * groups_0 + tl.arange(0, groups_0)) % out_channels
    offset_h = tl.arange(0, pool_size_0)[None, :] + pid_h * pool_stride_0 - pool_padding_0
    offset_w = tl.arange(0, pool_size_1)[:, None] + pid_w * pool_stride_1 - pool_padding_1
    offset_im_h = (pid_h * pool_stride_0 - pool_padding_0 + pool_dilation_0 * tl.arange(0, pool_size_0)[:, None]) * input_stride_2
    offset_im_w = (pid_w * pool_stride_1 - pool_padding_1 + pool_dilation_1 * tl.arange(0, pool_size_1)[None, :]) * input_stride_3
    offset_im = offset_n * input_stride_0 + offset_c[:, None] * input_stride_1 + offset_im_h + offset_im_w
    offset_w_1 = pid_c * weight_stride_0 + offset_c[:, None] * weight_stride_1 + (tl.arange(0, dilation_0)[:, None] * weight_stride_2 + tl.arange(0, dilation_1)[None, :] * weight_stride_3)
    w = tl.load(weight + offset_w_1)
    if bias is not None:
        offset_b = pid_c * bias_stride_0
        b = tl.load(bias + offset_b)
    else:
        b = 0
    mask_c = offset_c < out_channels
    mask_h = offset_h >= 0 & (offset_h < iH + padding_0 * 2)
    mask_w = offset_w >= 0 & (offset_w < iW + padding_1 * 2)
    mask_hw = mask_h[:, None] & mask_w[None, :]
    mask_im = offset_h[None, :] < iH + padding_0 * 2
    mask_im &= offset_w[:, None] < iW + padding_1 * 2
    mask = mask_c & mask_hw
    mask_im = mask_im & mask_hw
    a = tl.load(input + offset_im, mask=mask_im, other=0)
    a = tl.dot(w, a)
    a += b
    a = tl.where(mask, a, 0)
    offset_pool_h = pid_h * pool_stride_0 - pool_padding_0
    offset_pool_w = pid_w * pool_stride_1 - pool_padding_1
    offset_pool_h_1 = offset_pool_h + tl.arange(0, pool_size_0)
    offset_pool_w_1 = offset_pool_w + tl.arange(0, pool_size_1)
    mask_pool_h = offset_pool_h_1[:, None] >= 0 & (offset_pool_h_1[:, None] < iH + padding_0 * 2)
    mask_pool_w = offset_pool_w_1[None, :] >= 0 & (offset_pool_w_1[None, :] < iW + padding_1 * 2)
    mask_pool = mask_pool_h & mask_pool_w
    a = tl.where(mask_pool, a, float("-inf"))
    if pool_ceil_mode:
        c = tl.max(a, axis=(1, 2))
    else:
        c = tl.max(a, axis=(1, 2))
    offset_relu = offset_n * oH * oW + (pid_h * pool_size_0 + tl.arange(0, pool_size_0)[:, None]) * oW + (pid_w * pool_size_1 + tl.arange(0, pool_size_1)[None, :])
    if inplace:
        tl.store(input + offset_relu, c, mask=mask_c)
    else:
        offset_relu = offset_n * oH * oW + pid_h * pool_size_0 * oW + pid_w * pool_size_1 + tl.arange(0, pool_size_0)[:, None] * oW + tl.arange(0, pool_size_1)[None, :]
        tl.store(input + offset_relu, c, mask=mask_c)
    return

def relu_max_pool2d_conv2d(input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False):
    if input.dtype is torch.float16:
        return relu_max_pool2d_conv2d_triton(input, weight, bias, conv_stride, conv_padding, conv_dilation, conv_groups, pool_kernel_size, pool_stride, pool_padding, pool_dilation, pool_ceil_mode, inplace)
    else:
        raise Exception("only support float16")
