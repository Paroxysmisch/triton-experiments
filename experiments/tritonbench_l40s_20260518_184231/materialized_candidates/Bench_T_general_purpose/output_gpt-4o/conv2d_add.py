import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(input_ptr, weight_ptr, bias_ptr, other_ptr, out_ptr, 
                  minibatch, in_channels, iH, iW, out_channels, kH, kW, 
                  stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, 
                  groups, alpha, BLOCK_SIZE: tl.constexpr):
    # Calculate the indices for this block
    n = tl.program_id(0)  # minibatch index
    oc = tl.program_id(1)  # output channel index

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Compute convolution
    for ic in range(0, in_channels // groups):
        for kh in range(kH):
            for kw in range(kW):
                # Calculate input indices
                ih = tl.arange(0, BLOCK_SIZE) * stride_h + kh * dilation_h - pad_h
                iw = tl.arange(0, BLOCK_SIZE) * stride_w + kw * dilation_w - pad_w

                # Load input and weight
                input_val = tl.load(input_ptr + n * in_channels * iH * iW + ic * iH * iW + ih * iW + iw, mask=(ih >= 0) & (ih < iH) & (iw >= 0) & (iw < iW))
                weight_val = tl.load(weight_ptr + oc * (in_channels // groups) * kH * kW + ic * kH * kW + kh * kW + kw)

                # Accumulate result
                acc += input_val * weight_val

    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + oc)
        acc += bias_val

    # Add scaled 'other' if provided
    if other_ptr is not None:
        other_val = tl.load(other_ptr + n * out_channels * iH * iW + oc * iH * iW, mask=(ih >= 0) & (ih < iH) & (iw >= 0) & (iw < iW))
        acc += alpha * other_val

    # Store the result
    tl.store(out_ptr + n * out_channels * iH * iW + oc * iH * iW, acc)

import torch

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    # Extract dimensions
    minibatch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Calculate output dimensions
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h, pad_w = padding, padding
    elif padding == 'same':
        pad_h, pad_w = ((iH - 1) * stride_h + kH - iH) // 2, ((iW - 1) * stride_w + kW - iW) // 2
    elif padding == 'valid':
        pad_h, pad_w = 0, 0
    else:
        pad_h, pad_w = padding

    if isinstance(dilation, int):
        dilation_h, dilation_w = dilation, dilation
    else:
        dilation_h, dilation_w = dilation

    oH = (iH + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty((minibatch, out_channels, oH, oW), device=input.device, dtype=input.dtype)

    # Launch Triton kernel
    grid = (minibatch, out_channels)
    BLOCK_SIZE = 16  # Assume a block size for simplicity
    conv2d_kernel[grid](input, weight, bias, other, out,
                        minibatch, in_channels, iH, iW, out_channels, kH, kW,
                        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
                        groups, alpha, BLOCK_SIZE=BLOCK_SIZE)

    return out
