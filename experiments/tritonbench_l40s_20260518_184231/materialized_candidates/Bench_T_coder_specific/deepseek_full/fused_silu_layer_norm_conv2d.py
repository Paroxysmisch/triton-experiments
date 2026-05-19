import torch
import triton
import triton.language as tl

@triton.jit
def fused_silu_layer_norm_conv2d(
    x,
    weight,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    ln_eps,
):
    conv_out = tl.conv2d(
        x,
        conv_weight,
        conv_bias,
        stride=conv_stride,
        padding=conv_padding,
        dilation=conv_dilation,
        groups=conv_groups,
    )
    normalized_out = tl.layer_norm(conv_out, weight, ln_eps)
    output = tl.silu(normalized_out)
    return output

def test_fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias):
    triton_output = fused_silu_layer_norm_conv2d(
        x,
        weight,
        conv_weight,
        conv_bias,
        conv_stride=1,
        conv_padding=1,
        conv_dilation=1,
        conv_groups=1,
        ln_eps=1e-5,
    )
    return triton_output

# Example usage
x = torch.randn(4, 3, 32, 32)
conv_weight = torch.randn(8, 3, 3, 3)
conv_bias = torch.zeros(8)
weight = torch.ones(8)
output = test_fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias)
print(output.shape)
