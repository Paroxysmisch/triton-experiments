import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu_conv2d_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Matrix dimensions
    batch_size, in_channels, out_channels, in_height, in_width,
    kernel_height, kernel_width,
    # Parameters
    stride, padding, dilation, negative_slope,
    # Strides for the pointers
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    weight_out_stride, weight_in_stride, weight_height_stride, weight_width_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    BLOCK_SIZE: tl.constexpr
):
    # Compute output dimensions
    out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute batch and output channel indices
    batch_idx = pid // (out_channels * out_height * out_width)
    tmp = pid % (out_channels * out_height * out_width)
    out_channel = tmp // (out_height * out_width)
    tmp = tmp % (out_height * out_width)
    out_h = tmp // out_width
    out_w = tmp % out_width
    
    # Initialize accumulator
    acc = 0.0
    
    # Compute convolution
    for ic in range(in_channels):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                h_in = out_h * stride - padding + kh * dilation
                w_in = out_w * stride - padding + kw * dilation
                
                if 0 <= h_in < in_height and 0 <= w_in < in_width:
                    input_idx = (batch_idx * input_batch_stride +
                               ic * input_channel_stride +
                               h_in * input_height_stride +
                               w_in * input_width_stride)
                    weight_idx = (out_channel * weight_out_stride +
                                ic * weight_in_stride +
                                kh * weight_height_stride +
                                kw * weight_width_stride)
                    
                    input_val = tl.load(input_ptr + input_idx)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + out_channel)
    
    # Apply Leaky ReLU
    acc = tl.where(acc >= 0, acc, acc * negative_slope)
    
    # Store output
    output_idx = (batch_idx * output_batch_stride +
                 out_channel * output_channel_stride +
                 out_h * output_height_stride +
                 out_w * output_width_stride)
    tl.store(output_ptr + output_idx, acc)

def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False) -> torch.Tensor:
    """
    Applies 2D convolution followed by Leaky ReLU activation in one fused operation.
    
    Args:
        input: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Convolution weights of shape (out_channels, in_channels//groups, kernel_height, kernel_width)
        bias: Optional bias tensor of shape (out_channels)
        stride: Convolution stride. Default: 1
        padding: Convolution padding. Default: 0
        dilation: Convolution dilation. Default: 1
        groups: Number of groups for grouped convolution. Default: 1
        negative_slope: Negative slope for Leaky ReLU. Default: 0.01
        inplace: Whether to perform the operation in-place. Default: False
    
    Returns:
        Output tensor after convolution and Leaky ReLU activation
    """
    assert groups == 1, "Grouped convolution not yet supported in this implementation"
    
    # Get input dimensions
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Compute output dimensions
    out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), 
                        device=input.device, dtype=input.dtype)
    
    # Compute strides
    input_strides = input.stride()
    weight_strides = weight.stride()
    output_strides = output.stride()
    
    # Launch kernel
    grid = (batch_size * out_channels * out_height * out_width,)
    leaky_relu_conv2d_kernel[grid](
        input.data_ptr(), weight.data_ptr(),
        bias.data_ptr() if bias is not None else None,
        output.data_ptr(),
        batch_size, in_channels, out_channels, in_height, in_width,
        kernel_height, kernel_width,
        stride, padding, dilation, negative_slope,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3],
        weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3],
        BLOCK_SIZE=32
    )
    
    return output
