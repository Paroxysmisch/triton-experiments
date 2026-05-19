import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    in_channels, out_channels, kH, kW, iH, iW, oH, oW, groups,
    alpha, BLOCK_SIZE: tl.constexpr
):
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    h_idx = tl.program_id(2)
    w_idx = tl.arange(0, BLOCK_SIZE)

    # Calculate the starting position for the convolution
    h_start = h_idx * stride_h - pad_h
    w_start = w_idx * stride_w - pad_w

    # Initialize the result
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over the kernel dimensions
    for c in range(in_channels // groups):
        for kh in range(kH):
            for kw in range(kW):
                h_in = h_start + kh * dilation_h
                w_in = w_start + kw * dilation_w
                input_mask = (h_in >= 0) & (h_in < iH) & (w_in >= 0) & (w_in < iW)
                input_val = tl.load(
                    input_ptr + batch_id * in_channels * iH * iW +
                    (out_channel_id * (in_channels // groups) + c) * iH * iW +
                    h_in * iW + w_in, mask=input_mask, other=0.0
                )
                weight_val = tl.load(
                    weight_ptr + out_channel_id * (in_channels // groups) * kH * kW +
                    c * kH * kW + kh * kW + kw
                )
                result += input_val * weight_val

    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_id)
        result += bias_val

    # Add the other tensor or scalar, scaled by alpha
    if other_ptr is not None:
        other_val = tl.load(other_ptr + batch_id * out_channels * oH * oW +
                            out_channel_id * oH * oW + h_idx * oW + w_idx)
        result += alpha * other_val

    # Store the result
    tl.store(output_ptr + batch_id * out_channels * oH * oW +
             out_channel_id * oH * oW + h_idx * oW + w_idx, result)


def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    # Ensure input is on CUDA
    assert input.device.type == "cuda", "Input tensor must be on a CUDA device for triton ops."
    assert input.device == weight.device, "Input and weight must be on the same CUDA device."

    # Handle default values and parameter broadcasting
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    minibatch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    oH = (iH + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    oW = (iW + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

    if out is None:
        out = torch.empty((minibatch, out_channels, oH, oW), device=input.device, dtype=input.dtype)

    BLOCK_SIZE = 128
    grid = (minibatch, out_channels, oH)

    conv2d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias if bias is not None else 0,
        other_ptr=other if other is not None else 0,
        output_ptr=out,
        stride_h=stride[0], stride_w=stride[1],
        pad_h=padding[0], pad_w=padding[1],
        dilation_h=dilation[0], dilation_w=dilation[1],
        in_channels=in_channels, out_channels=out_channels,
        kH=kH, kW=kW, iH=iH, iW=iW, oH=oH, oW=oW, groups=groups,
        alpha=alpha,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
