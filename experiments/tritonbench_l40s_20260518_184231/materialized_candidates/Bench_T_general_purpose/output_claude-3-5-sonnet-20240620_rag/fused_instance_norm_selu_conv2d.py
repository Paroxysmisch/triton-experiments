import torch
import triton
import triton.language as tl
from typing import Optional, Union, Tuple
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def fused_conv2d_selu_instancenorm_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Matrix dimensions
    batch_size, in_channels, out_channels, in_height, in_width,
    kernel_height, kernel_width, out_height, out_width,
    # Convolution parameters
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
    # Instance norm parameters
    eps,
    # Strides
    input_batch_stride, input_channel_stride, input_height_stride,
    weight_output_stride, weight_input_stride, weight_height_stride,
    output_batch_stride, output_channel_stride, output_height_stride,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate batch and output channel indices
    batch_idx = pid // (out_channels * out_height * out_width)
    tmp = pid % (out_channels * out_height * out_width)
    out_channel_idx = tmp // (out_height * out_width)
    tmp = tmp % (out_height * out_width)
    out_h_idx = tmp // out_width
    out_w_idx = tmp % out_width
    
    # Compute input indices
    in_h_base = out_h_idx * stride_h - padding_h
    in_w_base = out_w_idx * stride_w - padding_w
    
    # Initialize accumulator
    acc = 0.0
    
    # Convolution
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            for ic in range(in_channels // groups):
                in_h = in_h_base + kh * dilation_h
                in_w = in_w_base + kw * dilation_w
                
                if 0 <= in_h < in_height and 0 <= in_w < in_width:
                    in_idx = (batch_idx * input_batch_stride +
                             ic * input_channel_stride +
                             in_h * input_height_stride + in_w)
                    weight_idx = (out_channel_idx * weight_output_stride +
                                ic * weight_input_stride +
                                kh * weight_height_stride + kw)
                    
                    x = tl.load(input_ptr + in_idx)
                    w = tl.load(weight_ptr + weight_idx)
                    acc += x * w
    
    # Add bias if present
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + out_channel_idx)
    
    # SELU activation
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    if acc <= 0:
        acc = scale * alpha * (tl.exp(acc) - 1.0)
    else:
        acc = scale * acc
    
    # Instance Normalization
    # Compute mean and variance
    mean = tl.sum(acc) / (out_height * out_width)
    var = tl.sum((acc - mean) ** 2) / (out_height * out_width)
    
    # Normalize
    out = (acc - mean) / tl.sqrt(var + eps)
    
    # Store result
    out_idx = (batch_idx * output_batch_stride +
               out_channel_idx * output_channel_stride +
               out_h_idx * output_height_stride + out_w_idx)
    tl.store(output_ptr + out_idx, out)

@torch.inference_mode()
def fused_instance_norm_selu_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    num_features: Optional[int] = None,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = False,
    track_running_stats: bool = False
) -> Tensor:
    """
    Applies a fused 2D convolution + SELU activation + instance normalization operation.
    
    Args:
        input (Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        weight (Tensor): Convolution weights
        bias (Optional[Tensor]): Optional bias tensor
        stride (int or tuple): Convolution stride. Default: 1
        padding (int or tuple): Convolution padding. Default: 0
        dilation (int or tuple): Convolution dilation. Default: 1
        groups (int): Number of groups for grouped convolution. Default: 1
        num_features (Optional[int]): Number of features for instance norm
        eps (float): Instance normalization epsilon. Default: 1e-5
        momentum (float): Instance normalization momentum. Default: 0.1
        affine (bool): If True, instance norm has learnable affine parameters
        track_running_stats (bool): If True, track running stats for instance norm
        
    Returns:
        Tensor: Output tensor after applying conv2d + SELU + instance norm
    """
    # Handle stride, padding, and dilation
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
        
    if isinstance(padding, int):
        padding_h = padding_w = padding
    else:
        padding_h, padding_w = padding
        
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    
    # Extract dimensions
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    out_height = ((in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) // stride_h) + 1
    out_width = ((in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) // stride_w) + 1
    
    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width),
                        device=input.device, dtype=input.dtype)
    
    # Calculate strides
    input_batch_stride = input.stride(0)
    input_channel_stride = input.stride(1)
    input_height_stride = input.stride(2)
    
    weight_output_stride = weight.stride(0)
    weight_input_stride = weight.stride(1)
    weight_height_stride = weight.stride(2)
    
    output_batch_stride = output.stride(0)
    output_channel_stride = output.stride(1)
    output_height_stride = output.stride(2)
    
    # Define grid and block sizes
    BLOCK_SIZE = 32
    grid = (batch_size * out_channels * out_height * out_width,)
    
    # Launch kernel
    fused_conv2d_selu_instancenorm_kernel[grid](
        input, weight,
        bias if bias is not None else torch.empty(0, device=input.device),
        output,
        batch_size, in_channels, out_channels, in_height, in_width,
        kernel_height, kernel_width, out_height, out_width,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
        eps,
        input_batch_stride, input_channel_stride, input_height_stride,
        weight_output_stride, weight_input_stride, weight_height_stride,
        output_batch_stride, output_channel_stride, output_height_stride,
        BLOCK_SIZE
    )
    
    return output
