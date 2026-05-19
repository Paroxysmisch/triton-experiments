import torch
import triton
import triton.language as tl
import math

@triton.jit
def sigmoid_conv2d_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Matrix dimensions
    batch, in_channels, out_channels, in_height, in_width,
    kernel_height, kernel_width,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    # Other parameters
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    # Position of elements
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    out_height = ((in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) // stride_h) + 1
    out_width = ((in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) // stride_w) + 1
    
    # Calculate batch and channel indices
    n = pid // (out_channels * out_height * out_width)
    c_out = (pid // (out_height * out_width)) % out_channels
    h_out = (pid // out_width) % out_height
    w_out = pid % out_width
    
    # Initialize accumulator
    acc = 0.0
    
    # Calculate group size
    channels_per_group = in_channels // groups
    group_idx = c_out // (out_channels // groups)
    
    # Convolution loop
    for ic in range(channels_per_group):
        c_in = group_idx * channels_per_group + ic
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                h_in = h_out * stride_h - padding_h + kh * dilation_h
                w_in = w_out * stride_w - padding_w + kw * dilation_w
                
                if 0 <= h_in < in_height and 0 <= w_in < in_width:
                    input_idx = (((n * in_channels + c_in) * in_height + h_in) * in_width + w_in)
                    weight_idx = (((c_out * channels_per_group + ic) * kernel_height + kh) * kernel_width + kw)
                    
                    inp_val = tl.load(input_ptr + input_idx)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    acc += inp_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + c_out)
    
    # Apply sigmoid activation
    acc = 1.0 / (1.0 + tl.exp(-acc))
    
    # Store result
    output_idx = (((n * out_channels + c_out) * out_height + h_out) * out_width + w_out)
    tl.store(output_ptr + output_idx, acc)

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    """
    Applies 2D convolution followed by sigmoid activation.
    
    Args:
        input (Tensor): Input tensor of shape (batch, in_channels, height, width)
        weight (Tensor): Filter tensor of shape (out_channels, in_channels/groups, kernel_height, kernel_width)
        bias (Tensor, optional): Bias tensor of shape (out_channels)
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int or tuple): Padding added to input. Default: 0
        dilation (int or tuple): Spacing between kernel elements. Default: 1
        groups (int): Number of blocked connections from input to output channels. Default: 1
        out (Tensor, optional): Output tensor
    
    Returns:
        Tensor: Output tensor after convolution and sigmoid activation
    """
    # Handle stride
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    # Handle padding
    if isinstance(padding, int):
        padding_h = padding_w = padding
    elif isinstance(padding, str):
        if padding.lower() == 'same':
            kernel_h, kernel_w = weight.shape[2:]
            padding_h = (kernel_h - 1) // 2
            padding_w = (kernel_w - 1) // 2
        elif padding.lower() == 'valid':
            padding_h = padding_w = 0
    else:
        padding_h, padding_w = padding
    
    # Handle dilation
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    
    # Get dimensions
    batch, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    out_height = ((in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) // stride_h) + 1
    out_width = ((in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) // stride_w) + 1
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty((batch, out_channels, out_height, out_width), 
                         device=input.device, dtype=input.dtype)
    
    # Launch kernel
    grid = (batch * out_channels * out_height * out_width,)
    sigmoid_conv2d_kernel[grid](
        input.contiguous(), weight.contiguous(),
        bias if bias is not None else None,
        out,
        batch, in_channels, out_channels,
        in_height, in_width,
        kernel_height, kernel_width,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        BLOCK_SIZE=32,
    )
    
    return out
