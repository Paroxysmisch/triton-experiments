import triton
import triton.language as tl

@triton.jit
def fused_silu_layer_norm_conv2d_kernel(
    X, W, B, LN_W, Y,
    stride, padding, dilation, groups,
    ln_eps, 
    OUT_H, OUT_W, 
    IN_C, IN_H, IN_W, 
    KERNEL_H, KERNEL_W, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_C: tl.constexpr, 
    BLOCK_SIZE_H: tl.constexpr, 
    BLOCK_SIZE_W: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute the output indices
    n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Mask out-of-bounds indices
    mask_n = n < OUT_H
    mask_c = c < OUT_W
    mask_h = h < IN_C
    mask_w = w < IN_H

    # Compute the input indices
    in_h = h * stride - padding
    in_w = w * stride - padding

    # Compute the convolution
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C), dtype=tl.float32)
    for kh in range(KERNEL_H):
        for kw in range(KERNEL_W):
            in_h_kh = in_h + kh * dilation
            in_w_kw = in_w + kw * dilation
            mask_h_kh = (in_h_kh >= 0) & (in_h_kh < IN_H)
            mask_w_kw = (in_w_kw >= 0) & (in_w_kw < IN_W)
            mask = mask_n & mask_c & mask_h & mask_w & mask_h_kh & mask_w_kw
            x = tl.load(X + (n[:, None] * IN_C * IN_H * IN_W + c[None, :] * IN_H * IN_W + in_h_kh[:, None] * IN_W + in_w_kw[None, :]), mask=mask, other=0.0)
            w = tl.load(W + (c[:, None] * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw), mask=mask, other=0.0)
            acc += x * w

    # Add bias if provided
    if B is not None:
        bias = tl.load(B + c, mask=mask_c, other=0.0)
        acc += bias

    # Layer normalization
    mean = tl.sum(acc, axis=1) / acc.shape[1]
    var = tl.sum((acc - mean[:, None]) ** 2, axis=1) / acc.shape[1]
    inv_std = 1.0 / tl.sqrt(var + ln_eps)
    normalized = (acc - mean[:, None]) * inv_std[:, None] * LN_W

    # SiLU activation
    y = normalized * tl.sigmoid(normalized)

    # Store the result
    tl.store(Y + (n[:, None] * OUT_W * IN_C * IN_H + c[None, :] * IN_H * IN_W + h[:, None] * IN_W + w[None, :]), y, mask=mask_n & mask_c & mask_h & mask_w)

import torch
import triton
import triton.language as tl

def fused_silu_layer_norm_conv2d(x: torch.Tensor, weight: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor = None, conv_stride: int = 1, conv_padding: int = 0, conv_dilation: int = 1, conv_groups: int = 1, ln_eps: float = 1e-5) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    x = x.cuda()
    conv_weight = conv_weight.cuda()
    if conv_bias is not None:
        conv_bias = conv_bias.cuda()
    weight = weight.cuda()

    # Compute the output dimensions
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_height, kernel_width = conv_weight.shape
    out_height = (in_height + 2 * conv_padding - conv_dilation * (kernel_height - 1) - 1) // conv_stride + 1
    out_width = (in_width + 2 * conv_padding - conv_dilation * (kernel_width - 1) - 1) // conv_stride + 1

    # Allocate the output tensor
    y = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)

    # Define the grid and block sizes
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_C = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    grid = (
        (out_height + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (out_width + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C,
        (in_channels + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (in_height + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )

    # Launch the Triton kernel
    fused_silu_layer_norm_conv2d_kernel[grid](
        x, conv_weight, conv_bias, weight, y,
        conv_stride, conv_padding, conv_dilation, conv_groups,
        ln_eps,
        out_height, out_width,
        in_channels, in_height, in_width,
        kernel_height, kernel_width,
        BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W
    )

    return y

# Example usage
if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 32)
    conv_weight = torch.randn(8, 3, 3, 3)
    conv_bias = torch.zeros(8)
    weight = torch.ones(8)
    output = fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias, conv_stride=1, conv_padding=1)
    print(output.shape)  # Expected: torch.Size([4, 8, 32, 32])
