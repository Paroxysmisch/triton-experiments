import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

# Triton kernel for 2D Convolution, Layer Normalization, and SiLU Activation
@triton.jit
def fused_silu_layer_norm_conv2d_kernel(
    x, conv_weight, conv_bias, weight, output,
    x_stride0, x_stride1, x_stride2, x_stride3,
    conv_weight_stride0, conv_weight_stride1, conv_weight_stride2, conv_weight_stride3,
    conv_bias_stride0,
    weight_stride0,
    output_stride0, output_stride1, output_stride2, output_stride3,
    N, C, H, W, K, R, S,
    stride, padding, dilation, groups,
    ln_eps,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // (H * W)
    h = (pid % (H * W)) // W
    w = (pid % (H * W)) % W

    # Compute the convolution output
    conv_output = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for r in range(R):
        for s in range(S):
            for k in range(K):
                for c in range(C // BLOCK_C):
                    x_offset = n * x_stride0 + c * x_stride1 + (h * stride + r - padding) * x_stride2 + (w * stride + s - padding) * x_stride3
                    conv_weight_offset = k * conv_weight_stride0 + c * conv_weight_stride1 + r * conv_weight_stride2 + s * conv_weight_stride3
                    x_val = tl.load(x + x_offset, mask=r < R and s < S and c < C // BLOCK_C, other=0.0)
                    conv_weight_val = tl.load(conv_weight + conv_weight_offset, mask=r < R and s < S and c < C // BLOCK_C, other=0.0)
                    conv_output += x_val * conv_weight_val

    # Add bias if provided
    if conv_bias is not None:
        bias_offset = k * conv_bias_stride0
        conv_output += tl.load(conv_bias + bias_offset, mask=k < K, other=0.0)

    # Layer Normalization
    mean = tl.sum(conv_output, axis=0) / K
    var = tl.sum((conv_output - mean) ** 2, axis=0) / K
    normalized_output = (conv_output - mean) / tl.sqrt(var + ln_eps)

    # Apply weight for Layer Normalization
    weight_offset = k * weight_stride0
    normalized_output *= tl.load(weight + weight_offset, mask=k < K, other=1.0)

    # Apply SiLU activation
    normalized_output = normalized_output * tl.sigmoid(normalized_output)

    # Store the output
    output_offset = n * output_stride0 + k * output_stride1 + h * output_stride2 + w * output_stride3
    tl.store(output + output_offset, normalized_output, mask=k < K)

# Wrapper function for the fused operation
@torch.inference_mode()
def fused_silu_layer_norm_conv2d(
    x: Tensor, weight: Tensor, conv_weight: Tensor, conv_bias: Tensor = None,
    conv_stride: int = 1, conv_padding: int = 0, conv_dilation: int = 1, conv_groups: int = 1, ln_eps: float = 1e-5
) -> Tensor:
    """
    Applies 2D Convolution, followed by Layer Normalization and SiLU activation to the input tensor `x`.

    Args:
        x (Tensor): Input tensor for convolution, normalization, and activation.
        weight (Tensor): Learnable weight of size matching normalized output dimensions for LayerNorm.
        conv_weight (Tensor): Convolution kernel tensor of appropriate dimensions.
        conv_bias (Tensor, optional): Convolution bias tensor. Default: ``None``.
        conv_stride (int, optional): Stride of convolution. Default: 1.
        conv_padding (int, optional): Padding added to both sides of input. Default: 0.
        conv_dilation (int, optional): Dilation of convolution kernel. Default: 1.
        conv_groups (int, optional): Number of groups for convolution. Default: 1.
        ln_eps (float, optional): Epsilon value for Layer Normalization. Default: 1e-5.

    Returns:
        Tensor: The output tensor after applying the fused operation.
    """
    N, C, H, W = x.shape
    K, _, R, S = conv_weight.shape
    out_channels = K
    out_height = (H + 2 * conv_padding - R) // conv_stride + 1
    out_width = (W + 2 * conv_padding - S) // conv_stride + 1

    # Output tensor
    output = torch.empty((N, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)

    # Block sizes
    BLOCK_N = 16
    BLOCK_C = 16
    BLOCK_H = 16
    BLOCK_W = 16

    # Grid size
    grid = (N * out_height * out_width,)

    # Launch the kernel
    fused_silu_layer_norm_conv2d_kernel[grid](
        x, conv_weight, conv_bias, weight, output,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2), conv_weight.stride(3),
        conv_bias.stride(0) if conv_bias is not None else 0,
        weight.stride(0),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        N, C, H, W, K, R, S,
        conv_stride, conv_padding, conv_dilation, conv_groups,
        ln_eps,
        BLOCK_N, BLOCK_C, BLOCK_H, BLOCK_W,
        num_warps=4,
        num_stages=2,
    )

    return output

# Example usage
x = torch.randn(4, 3, 32, 32)
conv_weight = torch.randn(8, 3, 3, 3)
conv_bias = torch.zeros(8)
weight = torch.ones(8)
output = fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias, conv_stride=1, conv_padding=1)
print(output.shape)  # Expected output shape: torch.Size([4, 8, 32, 32])
