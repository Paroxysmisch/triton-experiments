import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def fused_conv2d_selu_instance_norm_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    stride, padding, dilation, groups,
    num_features, eps, affine,
    N: tl.constexpr, M: tl.constexpr, K: tl.constexpr
):
    # Kernel implementation goes here
    # This is a placeholder for the actual Triton kernel logic
    # Load input, weight, and bias
    # Perform convolution
    # Apply SELU activation
    # Perform instance normalization
    # Store the result in output

@torch.inference_mode()
def fused_instance_norm_selu_conv2d(
    input: Tensor, weight: Tensor, bias=None,
    stride=1, padding=0, dilation=1, groups=1,
    num_features=None, eps=1e-5, momentum=0.1,
    affine=False, track_running_stats=False
) -> Tensor:
    """
    Applies a fused operation consisting of a 2D convolution followed by SELU activation and instance normalization.

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
    # Determine the output shape
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Calculate output dimensions based on convolution parameters
    oH = (iH + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    oW = (iW + 2 * padding - dilation * (kW - 1) - 1) // stride + 1

    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, oH, oW), device=input.device, dtype=input.dtype)

    # Kernel grid and meta information
    grid = (batch_size * out_channels, oH, oW)
    kernel_meta = dict(device=input.device, stream=get_cuda_stream(input.device.index))

    # Launch Triton kernel
    fused_conv2d_selu_instance_norm_kernel[grid](
        input, weight, bias, output,
        stride, padding, dilation, groups,
        num_features, eps, affine,
        N=in_channels, M=out_channels, K=kH*kW,
        num_warps=4, num_stages=2,
        **kernel_meta
    )

    return output
