import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def layer_norm_silu_kernel(
    input_ptr,
    gamma_ptr,
    output_ptr,
    n_channels,
    eps,
    input_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    channel_offsets = tl.arange(0, BLOCK_SIZE)
    # Load input row
    input_row_ptr = input_ptr + row_idx * input_row_stride
    input_row = tl.load(input_row_ptr + channel_offsets, mask=channel_offsets < n_channels, other=0.0)
    input_float = input_row.to(tl.float32)
    # Compute mean and variance
    mean = tl.sum(input_float, axis=0) / n_channels
    centered = input_float - mean
    var = tl.sum(centered * centered, axis=0) / n_channels
    std = tl.sqrt(var + eps)
    # Normalize and scale with gamma
    normalized = centered / std
    gamma = tl.load(gamma_ptr + channel_offsets, mask=channel_offsets < n_channels, other=1.0)
    scaled = normalized * gamma.to(tl.float32)
    # Apply SiLU activation
    silu = scaled * tl.sigmoid(scaled)
    # Store output
    output_row_ptr = output_ptr + row_idx * input_row_stride
    tl.store(output_row_ptr + channel_offsets, silu.to(input_row.dtype), mask=channel_offsets < n_channels)

@torch.inference_mode()
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
    # Apply convolution
    conv_out = torch.nn.functional.conv2d(
        x,
        conv_weight,
        conv_bias,
        stride=conv_stride,
        padding=conv_padding,
        dilation=conv_dilation,
        groups=conv_groups,
    )
    # Prepare for LayerNorm + SiLU
    N, C, H, W = conv_out.shape
    input_2d = conv_out.permute(0, 2, 3, 1).contiguous().view(-1, C)
    output_2d = torch.empty_like(input_2d)
    # Kernel configuration
    BLOCK_SIZE = triton.next_power_of_2(C)
    grid = (input_2d.size(0),)
    # Get CUDA stream for kernel execution
    device = x.device
    stream = get_cuda_stream(device.index)
    # Launch kernel
    layer_norm_silu_kernel[grid](
        input_2d,
        weight,
        output_2d,
        C,
        ln_eps,
        input_2d.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
        stream=stream,
    )
    # Reshape output to original dimensions
    output = output_2d.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
    return output
