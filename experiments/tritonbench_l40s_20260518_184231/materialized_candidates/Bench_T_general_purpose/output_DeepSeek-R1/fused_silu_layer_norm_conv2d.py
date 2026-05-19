import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _layer_norm_silu_kernel(
    x_ptr,              # Pointer to convolution output
    y_ptr,              # Pointer to output tensor
    weight_ptr,         # Pointer to layer norm weight tensor
    C,                  # Number of channels
    N, H, W,            # Other dimensions (batch, height, width)
    eps,                # Epsilon for layer norm
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    if c >= C:
        return

    # Pointers to the start of the current channel in input and output
    x_channel_ptr = x_ptr + c * N * H * W
    y_channel_ptr = y_ptr + c * N * H * W

    # Compute mean and variance for the channel
    total_elements = N * H * W
    sum = tl.zeros((1,), tl.float32)
    sum_sq = tl.zeros((1,), tl.float32)

    for idx in range(0, total_elements, BLOCK_SIZE):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        # Load data
        x_val = tl.load(x_channel_ptr + offsets, mask=mask, other=0.0)
        x_val_float = x_val.to(tl.float32)
        sum += tl.sum(x_val_float, axis=0)
        sum_sq += tl.sum(x_val_float * x_val_float, axis=0)

    # Block-wide reduction
    total_sum = tl.sum(sum)
    total_sum_sq = tl.sum(sum_sq)
    mean = total_sum / total_elements
    var = (total_sum_sq / total_elements) - (mean * mean)
    std = tl.sqrt(var + eps)

    # Load weight for this channel
    w = tl.load(weight_ptr + c)

    # Normalize and apply SiLU
    for idx in range(0, total_elements, BLOCK_SIZE):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        x_val = tl.load(x_channel_ptr + offsets, mask=mask, other=0.0)
        x_val_float = x_val.to(tl.float32)
        x_norm = (x_val_float - mean) / std * w
        silu = x_norm * tl.sigmoid(x_norm)

        # Store back to output
        tl.store(y_channel_ptr + offsets, silu.to(x_val.dtype), mask=mask)

def fused_silu_layer_norm_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor = None,
    conv_stride: int = 1,
    conv_padding: int = 0,
    conv_dilation: int = 1,
    conv_groups: int = 1,
    ln_eps: float = 1e-5,
) -> torch.Tensor:
    # Perform convolution
    conv_out = F.conv2d(
        x,
        conv_weight,
        bias=conv_bias,
        stride=conv_stride,
        padding=conv_padding,
        dilation=conv_dilation,
        groups=conv_groups,
    ).contiguous()

    # Ensure weight is correctly shaped
    C = conv_out.size(1)
    if weight.size(0) != C:
        raise ValueError(f"LayerNorm weight must have {C} elements, got {weight.size(0)}")

    # Prepare output tensor
    y = torch.empty_like(conv_out)

    # Launch kernel
    N, C, H, W = conv_out.shape
    BLOCK_SIZE = 128  # Tune this for optimal performance

    grid = (C,)
    _layer_norm_silu_kernel[grid](
        conv_out, y, weight,
        C, N, H, W, ln_eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y
