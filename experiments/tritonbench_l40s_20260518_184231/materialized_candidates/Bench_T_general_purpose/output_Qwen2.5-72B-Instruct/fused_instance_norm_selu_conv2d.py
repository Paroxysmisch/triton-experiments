import triton
import triton.language as tl

@triton.jit
def fused_instance_norm_selu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    stride, padding, dilation, groups,
    num_features, eps, momentum, affine, track_running_stats,
    input_shape, weight_shape, output_shape,
    BLOCK_SIZE: tl.constexpr
):
    # Load parameters
    minibatch, in_channels, iH, iW = input_shape
    out_channels, _, kH, kW = weight_shape
    _, _, oH, oW = output_shape

    # Compute the output indices
    pid = tl.program_id(axis=0)
    n = pid // (oH * oW)
    h = (pid % (oH * oW)) // oW
    w = (pid % (oH * oW)) % oW

    # Compute the input indices
    ho = h * stride - padding
    wo = w * stride - padding

    # Initialize the output value
    output_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the kernel
    for kh in range(kH):
        for kw in range(kW):
            hi = ho + kh * dilation
            wi = wo + kw * dilation
            if 0 <= hi < iH and 0 <= wi < iW:
                for c in range(in_channels // groups):
                    input_val = tl.load(input_ptr + n * in_channels * iH * iW + (c + (in_channels // groups) * (pid % (oH * oW))) * iH * iW + hi * iW + wi)
                    weight_val = tl.load(weight_ptr + (c + (in_channels // groups) * (pid % (oH * oW))) * kH * kW + kh * kW + kw)
                    output_val += input_val * weight_val

    # Apply bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + (pid % (oH * oW)))
        output_val += bias_val

    # Apply SELU activation
    output_val = 1.0507009873554804934193349852946 * (tl.where(output_val > 0, output_val, 1.6732632423543772848170429916717 * (tl.exp(output_val) - 1)))

    # Instance normalization
    if num_features is not None:
        mean = tl.sum(output_val) / num_features
        var = tl.sum((output_val - mean) ** 2) / num_features
        output_val = (output_val - mean) / tl.sqrt(var + eps)

        if affine:
            gamma = tl.load(gamma_ptr + (pid % (oH * oW)))
            beta = tl.load(beta_ptr + (pid % (oH * oW)))
            output_val = gamma * output_val + beta

    # Store the result
    tl.store(output_ptr + n * out_channels * oH * oW + (pid % (oH * oW)), output_val)

import torch
import triton
import triton.language as tl

def fused_instance_norm_selu_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, stride=1, padding=0, dilation=1, groups=1, num_features=None, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False) -> torch.Tensor:
    # Input shape: (minibatch, in_channels, iH, iW)
    # Weight shape: (out_channels, in_channels // groups, kH, kW)
    # Output shape: (minibatch, out_channels, oH, oW)

    minibatch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    # Calculate output dimensions
    oH = (iH + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    oW = (iW + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

    # Allocate output tensor
    output = torch.empty((minibatch, out_channels, oH, oW), device=input.device, dtype=input.dtype)

    # Launch the kernel
    grid = (minibatch * oH * oW, )
    fused_instance_norm_selu_conv2d_kernel[grid](
        input, weight, bias, output,
        stride[0], padding[0], dilation[0], groups,
        num_features, eps, momentum, affine, track_running_stats,
        (minibatch, in_channels, iH, iW), (out_channels, in_channels // groups, kH, kW), (minibatch, out_channels, oH, oW),
        BLOCK_SIZE=128
    )

    return output
