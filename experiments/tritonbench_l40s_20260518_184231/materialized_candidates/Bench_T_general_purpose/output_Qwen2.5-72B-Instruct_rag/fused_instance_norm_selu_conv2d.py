import torch
import triton
import triton.language as tl

@triton.jit
def fused_instance_norm_selu_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_channels,
    out_channels,
    in_h,
    in_w,
    out_h,
    out_w,
    k_h,
    k_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    num_features,
    eps,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE_H * BLOCK_SIZE_W
    num_blocks = (out_h * out_w + block_size - 1) // block_size
    block_id = pid % num_blocks
    block_h = block_id // (out_w // BLOCK_SIZE_W)
    block_w = block_id % (out_w // BLOCK_SIZE_W)

    # Compute the starting and ending indices for the block
    start_h = block_h * BLOCK_SIZE_H
    start_w = block_w * BLOCK_SIZE_W
    end_h = min(start_h + BLOCK_SIZE_H, out_h)
    end_w = min(start_w + BLOCK_SIZE_W, out_w)

    # Compute the starting and ending indices for the input
    start_h_in = start_h * stride_h - padding_h
    start_w_in = start_w * stride_w - padding_w
    end_h_in = start_h_in + (end_h - start_h) * stride_h + (k_h - 1) * dilation_h
    end_w_in = start_w_in + (end_w - start_w) * stride_w + (k_w - 1) * dilation_w

    # Load the input and weight tensors
    input_block = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, in_channels), dtype=tl.float32)
    weight_block = tl.load(weight_ptr, mask=tl.arange(0, out_channels) < out_channels)

    for h in range(start_h, end_h):
        for w in range(start_w, end_w):
            for c in range(0, in_channels, BLOCK_SIZE_C):
                input_block[h - start_h, w - start_w, c:c + BLOCK_SIZE_C] = tl.load(
                    input_ptr + (h * stride_h - padding_h) * in_w * in_channels + (w * stride_w - padding_w) * in_channels + c,
                    mask=(h * stride_h - padding_h) * in_w * in_channels + (w * stride_w - padding_w) * in_channels + c < in_h * in_w * in_channels
                )

    # Perform the convolution
    conv_output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, out_channels), dtype=tl.float32)
    for c in range(0, in_channels, BLOCK_SIZE_C):
        for kh in range(k_h):
            for kw in range(k_w):
                conv_output += tl.dot(
                    input_block[:, :, c:c + BLOCK_SIZE_C],
                    weight_block[kh * k_w * in_channels + kw * in_channels + c:c + BLOCK_SIZE_C, :]
                )

    # Apply bias if provided
    if bias_ptr is not None:
        bias_block = tl.load(bias_ptr, mask=tl.arange(0, out_channels) < out_channels)
        conv_output += bias_block

    # Apply SELU activation
    conv_output = 1.0507 * tl.where(conv_output > 0, conv_output, 1.67326 * tl.exp(conv_output) - 1.67326)

    # Compute instance normalization
    mean = tl.sum(conv_output, axis=2) / out_channels
    var = tl.sum((conv_output - mean[:, :, None]) ** 2, axis=2) / out_channels
    conv_output = (conv_output - mean[:, :, None]) / tl.sqrt(var[:, :, None] + eps)

    # Apply affine transformation if affine is True
    if affine:
        gamma = tl.load(gamma_ptr, mask=tl.arange(0, out_channels) < out_channels)
        beta = tl.load(beta_ptr, mask=tl.arange(0, out_channels) < out_channels)
        conv_output = conv_output * gamma + beta

    # Store the output
    for h in range(start_h, end_h):
        for w in range(start_w, end_w):
            tl.store(
                output_ptr + h * out_w * out_channels + w * out_channels,
                conv_output[h - start_h, w - start_w, :],
                mask=(h < out_h) & (w < out_w)
            )

import torch
from triton.runtime.jit import get_cuda_stream

def fused_instance_norm_selu_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int or tuple = 1,
    padding: int or tuple = 0,
    dilation: int or tuple = 1,
    groups: int = 1,
    num_features: int = None,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = False,
    track_running_stats: bool = False
) -> torch.Tensor:
    """
    Applies a fused operation consisting of a 2D convolution followed by SELU activation and instance normalization on the input tensor.

    Args:
        input (Tensor): Input tensor of shape (minibatch, in_channels, iH, iW).
        weight (Tensor): Weights for the convolution, shape (out_channels, in_channels / groups, kH, kW).
        bias (Tensor, optional): Bias for the convolution layer, shape (out_channels).
        stride (int or tuple, optional): Stride of the convolution. Default is 1.
        padding (int or tuple, optional): Padding for the convolution. Default is 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default is 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
        num_features (int, optional): Number of features or channels in the input for instance normalization.
        eps (float, optional): A value added to the denominator for numerical stability in instance normalization. Default is 1e-5.
        momentum (float, optional): Momentum for updating running statistics in instance normalization. Default is 0.1.
        affine (bool, optional): If True, instance normalization has learnable affine parameters. Default is False.
        track_running_stats (bool, optional): If True, tracks running mean and variance for instance normalization. Default is False.

    Returns:
        Tensor: The output tensor after applying the fused operation.
    """

    # Ensure input and weight are on the same device
    device = input.device
    weight = weight.to(device)
    if bias is not None:
        bias = bias.to(device)

    # Get input and weight dimensions
    minibatch, in_channels, in_h, in_w = input.shape
    out_channels, in_channels_per_group, k_h, k_w = weight.shape

    # Compute output dimensions
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        padding_h, padding_w = padding, padding
    else:
        padding_h, padding_w = padding

    if isinstance(dilation, int):
        dilation_h, dilation_w = dilation, dilation
    else:
        dilation_h, dilation_w = dilation

    out_h = (in_h + 2 * padding_h - dilation_h * (k_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * padding_w - dilation_w * (k_w - 1) - 1) // stride_w + 1

    # Allocate output tensor
    output = torch.empty((minibatch, out_channels, out_h, out_w), device=device)

    # Set up grid and block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32

    grid = (minibatch, (out_h * out_w + BLOCK_SIZE_H * BLOCK_SIZE_W - 1) // (BLOCK_SIZE_H * BLOCK_SIZE_W))

    # Launch the kernel
    fused_instance_norm_selu_conv2d_kernel[grid](
        input,
        weight,
        bias,
        output,
        in_channels,
        out_channels,
        in_h,
        in_w,
        out_h,
        out_w,
        k_h,
        k_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        num_features,
        eps,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
        BLOCK_SIZE_C,
        num_warps=4,
        num_stages=2,
        stream=get_cuda_stream(device.index)
    )

    return output
