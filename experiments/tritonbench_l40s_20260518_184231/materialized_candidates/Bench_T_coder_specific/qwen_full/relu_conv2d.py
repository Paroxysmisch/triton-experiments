import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import relu

@triton.jit
def relu_conv2d_kernel(
    input, weight, bias, stride, padding, output, n, c, h, w, k, Y, X, stride_n, stride_c, stride_h, stride_w, stride_k, stride_Y, stride_X, stride_bias, S, P, GROUPS, inplace: tl.constexpr
):
    pid = tl.program_id(0)
    # compute output index
    y = tl.arange(0, Y)[None, :] + pid * stride_Y
    x = tl.arange(0, X)[:, None] + tl.program_id(1) * stride_X
    n_idx = (y // S) * P + (x // P)
    c_idx = ((y % S) * P + (x % P)) // (h * w)
    k_idx = ((y % S) * P + (x % P)) % (h * w)
    # compute input index
    input_n_idx = n_idx // stride_n
    input_c_idx = c_idx * GROUPS
    input_h_idx = (k_idx // stride_w) % h
    input_w_idx = k_idx % w
    # compute weight index
    weight_c_idx = c_idx
    weight_k_idx = k_idx
    # compute
    if inplace:
        input_ptrs = input + input_n_idx * stride_n + (input_c_idx + tl.arange(0, GROUPS)) * stride_c + input_h_idx * stride_h + input_w_idx * stride_w
        input_val = tl.load(input_ptrs, mask=(input_n_idx < n) & (input_c_idx + tl.arange(0, GROUPS) < c) & (input_h_idx < h) & (input_w_idx < w), other=0)
        weight_ptrs = weight + weight_c_idx * stride_k + weight_k_idx
        weight_val = tl.load(weight_ptrs, mask=(weight_c_idx < c // GROUPS) & (weight_k_idx < k), other=0)
        acc = tl.sum(input_val * weight_val, 0)
    else:
        acc = tl.zeros([1], dtype=tl.float32)
        input_ptrs = input + input_n_idx * stride_n + input_c_idx * stride_c + (input_h_idx * stride_h + input_w_idx * stride_w)[None, :]
        weight_ptrs = weight + weight_c_idx * stride_k + weight_k_idx[:, None]
        mask_n = input_n_idx < n
        mask_c = (input_c_idx + tl.arange(0, GROUPS)) < c
        mask_k = weight_k_idx < k
        mask_h = input_h_idx < h
        mask_w = input_w_idx < w
        mask = mask_n & mask_c & (mask_k[:, None] & mask_h & mask_w)
        input_val = tl.load(input_ptrs, mask=mask, other=0)
        weight_val = tl.load(weight_ptrs, mask=mask_k[:, None] & mask_k[:, None], other=0)
        acc += tl.sum(input_val * weight_val, 0)
    if bias is not None:
        bias_ptrs = bias + c_idx * stride_bias
        bias_val = tl.load(bias_ptrs, mask=(c_idx < c // GROUPS), other=0)
        acc += bias_val
    acc = relu(acc)
    output_ptrs = output + n_idx * stride_n + (c_idx + tl.arange(0, GROUPS)) * stride_c + input_h_idx * stride_h + input_w_idx * stride_w
    tl.store(output_ptrs, acc, mask=(n_idx < n) & (c_idx + tl.arange(0, GROUPS) < c) & (input_h_idx < h) & (input_w_idx < w))


def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    if type(stride) == int:
        stride = (stride, stride)
    if type(padding) == int:
        padding = (padding, padding)
    if type(dilation) == int:
        dilation = (dilation, dilation)
    batch, in_channels, height, width = input.shape
    out_channels, channels, kernel_height, kernel_width = weight.shape
    assert channels * groups == in_channels
    assert out_channels % groups == 0
    out_channels_per_group = out_channels // groups
    assert (kernel_height == 1) & (kernel_width == 1)
    assert (height - kernel_height + 2 * padding[0]) % stride[0] == 0
    assert (width - kernel_width + 2 * padding[1]) % stride[1] == 0
    output_height = (height - kernel_height + 2 * padding[0]) // stride[0] + 1
    output_width = (width - kernel_width + 2 * padding[1]) // stride[1] + 1
    output = torch.empty((batch, out_channels, output_height, output_width), dtype=input.dtype, device=input.device)
    grid = lambda META: (triton.cdiv(output_height, META["Y"]), triton.cdiv(output_width, META["X"]))
    with torch.cuda.device(input.device):
        relu_conv2d_kernel[grid](
            input,
            weight,
            bias,
            stride,
            padding,
            output,
            batch,
            in_channels,
            height,
            width,
            out_channels_per_group,
            output_height,
            output_width,
            batch.stride(0),
            batch.stride(1),
            batch.stride(2),
            batch.stride(3),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            weight.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            bias.stride(0) if bias is not None else 0,
            stride[0],
            padding[0],
            groups,
            inplace,
        )
    return output
