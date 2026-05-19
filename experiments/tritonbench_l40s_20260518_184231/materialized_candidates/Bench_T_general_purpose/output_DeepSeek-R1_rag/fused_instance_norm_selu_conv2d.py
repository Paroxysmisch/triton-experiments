import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn.functional import conv2d
from triton.runtime.jit import get_cuda_stream

@triton.jit
def fused_selu_instance_norm_kernel(
    # Input and output pointers
    conv_output_ptr,
    output_ptr,
    # Instance normalization parameters
    num_features,
    eps,
    # Tensor dimensions
    batch_size,
    in_channels,
    in_h,
    in_w,
    # Convolution parameters
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    # Memory strides for conv_output
    conv_output_batch_stride,
    conv_output_channel_stride,
    conv_output_h_stride,
    conv_output_w_stride,
    # Memory strides for output
    output_batch_stride,
    output_channel_stride,
    output_h_stride,
    output_w_stride,
    # Triton constants
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get the program ID for the current instance (batch and channel)
    pid = tl.program_id(0)
    num_instances = batch_size * num_features
    if pid >= num_instances:
        return

    batch_idx = pid // num_features
    channel_idx = pid % num_features

    # Calculate the start of the current feature map in conv_output and output
    conv_output_feature_map_ptr = (
        conv_output_ptr
        + batch_idx * conv_output_batch_stride
        + channel_idx * conv_output_channel_stride
    )
    output_feature_map_ptr = (
        output_ptr
        + batch_idx * output_batch_stride
        + channel_idx * output_channel_stride
    )

    # Initialize sum and sum of squares for instance normalization
    sum_selu = 0.0
    sum_sq_selu = 0.0

    # Loop over the spatial dimensions (h, w) in blocks
    for h in range(0, in_h, BLOCK_SIZE_H):
        h_offsets = h + tl.arange(0, BLOCK_SIZE_H)
        h_mask = h_offsets < in_h
        for w in range(0, in_w, BLOCK_SIZE_W):
            w_offsets = w + tl.arange(0, BLOCK_SIZE_W)
            w_mask = w_offsets < in_w
            # Generate a 2D mask for valid (h, w) positions
            mask = h_mask[:, None] & w_mask[None, :]
            # Load the input block
            input_ptr = conv_output_feature_map_ptr + h_offsets[:, None] * conv_output_h_stride + w_offsets[None, :] * conv_output_w_stride
            x = tl.load(input_ptr, mask=mask, other=0.0)
            x_float = x.to(tl.float32)
            # Apply SELU activation
            alpha = 1.67326
            scale = 1.0507
            selu_x = tl.where(x_float > 0, scale * x_float, scale * alpha * (tl.exp(x_float) - 1))
            # Accumulate sum and sum of squares
            sum_selu += tl.sum(selu_x)
            sum_sq_selu += tl.sum(selu_x * selu_x)

    # Compute mean and variance
    num_elements = in_h * in_w
    mean = sum_selu / num_elements
    variance = (sum_sq_selu / num_elements) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(variance + eps)

    # Normalize and store the result
    for h in range(0, in_h, BLOCK_SIZE_H):
        h_offsets = h + tl.arange(0, BLOCK_SIZE_H)
        h_mask = h_offsets < in_h
        for w in range(0, in_w, BLOCK_SIZE_W):
            w_offsets = w + tl.arange(0, BLOCK_SIZE_W)
            w_mask = w_offsets < in_w
            mask = h_mask[:, None] & w_mask[None, :]
            input_ptr = conv_output_feature_map_ptr + h_offsets[:, None] * conv_output_h_stride + w_offsets[None, :] * conv_output_w_stride
            x = tl.load(input_ptr, mask=mask, other=0.0)
            x_float = x.to(tl.float32)
            # Apply SELU activation again to avoid storing intermediates
            selu_x = tl.where(x_float > 0, scale * x_float, scale * alpha * (tl.exp(x_float) - 1))
            # Normalize
            normalized = (selu_x - mean) * inv_std
            # Store the result
            output_ptr = output_feature_map_ptr + h_offsets[:, None] * output_h_stride + w_offsets[None, :] * output_w_stride
            tl.store(output_ptr, normalized.to(x.dtype), mask=mask)

@torch.inference_mode()
def fused_instance_norm_selu_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    stride: int or tuple = 1,
    padding: int or tuple = 0,
    dilation: int or tuple = 1,
    groups: int = 1,
    num_features: int = None,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = False,
    track_running_stats: bool = False,
) -> Tensor:
    # Perform convolution
    output = conv2d(input, weight, bias, stride, padding, dilation, groups)
    batch_size, out_channels, out_h, out_w = output.shape

    # Validate num_features
    if num_features is not None:
        if num_features != out_channels:
            raise ValueError(f"num_features must be equal to the number of output channels ({out_channels})")
    else:
        num_features = out_channels

    # Handle instance normalization parameters (affine not supported in this implementation)
    if affine:
        raise NotImplementedError("Affine transformation for instance normalization is not supported in this fused kernel")

    # Reshape output to (batch_size, num_features, out_h, out_w)
    output = output.contiguous()

    # Prepare output tensor
    normalized_output = torch.empty_like(output)

    # Launch kernel
    def _kernel_meta():
        device = output.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    grid = (batch_size * out_channels,)

    fused_selu_instance_norm_kernel[grid](
        output,
        normalized_output,
        num_features,
        eps,
        batch_size,
        out_channels,
        out_h,
        out_w,
        stride if isinstance(stride, int) else stride[0],
        stride if isinstance(stride, int) else stride[1],
        padding if isinstance(padding, int) else padding[0],
        padding if isinstance(padding, int) else padding[1],
        dilation if isinstance(dilation, int) else dilation[0],
        dilation if isinstance(dilation, int) else dilation[1],
        groups,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        normalized_output.stride(0),
        normalized_output.stride(1),
        normalized_output.stride(2),
        normalized_output.stride(3),
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        num_warps=4,
        num_stages=2,
        **_kernel_meta(),
    )

    return normalized_output
