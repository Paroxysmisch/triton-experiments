import torch
import triton
import triton.language as tl
from torch.nn.functional import conv2d, batch_norm

@triton.jit
def relu_batch_norm_conv2d_kernel(
    # Input tensors
    input_ptr, weight_ptr,
    # Output tensor
    output_ptr,
    # Convolution parameters
    in_channels, out_channels, iH, iW, kH, kW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups,
    # Batch norm parameters
    running_mean_ptr, running_var_ptr,
    bn_weight_ptr, bn_bias_ptr,
    # Training mode
    training,
    # BN hyperparams
    momentum, eps,
    # Tensor metadata
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    weight_outc_stride, weight_inc_stride, weight_h_stride, weight_w_stride,
    output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
    # Blocking
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid over batch, output channels, and output spatial dimensions
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute output spatial dimensions
    oH = (iH + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1

    # Input pointers for current batch
    input_ptr += pid_batch * input_batch_stride
    output_ptr += pid_batch * output_batch_stride

    # Initialize convolution accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Loop over kernel elements and input channels
    for kh in range(kH):
        for kw in range(kW):
            for ic in range(in_channels // groups):
                # Calculate input position with dilation and padding
                h = pid_h * stride_h - pad_h + kh * dilation_h
                w = pid_w * stride_w - pad_w + kw * dilation_w

                # Check if input position is valid
                if h >= 0 and h < iH and w >= 0 and w < iW:
                    input_offset = ic * input_channel_stride + h * input_h_stride + w * input_w_stride
                    weight_offset = (pid_channel * groups * (in_channels // groups) + ic) * weight_inc_stride + kh * weight_h_stride + kw * weight_w_stride

                    # Load input and weight values
                    input_val = tl.load(input_ptr + input_offset)
                    weight_val = tl.load(weight_ptr + weight_offset)

                    acc += tl.dot(input_val, weight_val, allow_tf32=True)

    # Load batch norm parameters
    if bn_weight_ptr is not None:
        gamma = tl.load(bn_weight_ptr + pid_channel)
    else:
        gamma = 1.0
        
    if bn_bias_ptr is not None:
        beta = tl.load(bn_bias_ptr + pid_channel)
    else:
        beta = 0.0

    # Batch normalization
    if training:
        # Compute mean and variance for current channel
        mean = tl.sum(acc) / (oH * oW)
        var = tl.sum((acc - mean)**2) / (oH * oW)
        
        # Update running stats
        if running_mean_ptr is not None and running_var_ptr is not None:
            running_mean = tl.load(running_mean_ptr + pid_channel)
            running_var = tl.load(running_var_ptr + pid_channel)
            
            new_mean = (1 - momentum) * running_mean + momentum * mean
            new_var = (1 - momentum) * running_var + momentum * var
            
            tl.store(running_mean_ptr + pid_channel, new_mean)
            tl.store(running_var_ptr + pid_channel, new_var)
    else:
        mean = tl.load(running_mean_ptr + pid_channel)
        var = tl.load(running_var_ptr + pid_channel)

    # Apply batch norm
    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized = (acc - mean) * gamma * inv_std + beta

    # Apply ReLU
    output = tl.maximum(normalized, 0.0)

    # Store result
    output_offset = pid_channel * output_channel_stride + pid_h * output_h_stride + pid_w * output_w_stride
    tl.store(output_ptr + output_offset, output)

def relu_batch_norm_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    running_mean=None,
    running_var=None,
    bn_weight=None,
    bn_bias=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
    inplace=False
) -> torch.Tensor:
    # Validate inputs
    assert input.dim() == 4, "Input must be 4D (batch, channels, height, width)"
    assert weight.dim() == 4, "Weight must be 4D (out_c, in_c/groups, kH, kW)"
    
    # Convert stride/padding/dilation to tuples
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    # Get output shape
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    oH = (iH + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    oW = (iW + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

    # Allocate output tensor
    output = torch.empty((batch_size, out_channels, oH, oW), 
                        device=input.device, dtype=input.dtype)

    # Launch kernel
    grid = lambda meta: (batch_size, out_channels, oH, oW)
    
    relu_batch_norm_conv2d_kernel[grid](
        input, weight, output,
        # Convolution parameters
        in_channels, out_channels, iH, iW, kH, kW,
        stride[0], stride[1], padding[0], padding[1], 
        dilation[0], dilation[1], groups,
        # BN parameters
        running_mean, running_var, bn_weight, bn_bias,
        # Training mode
        training,
        # BN hyperparams
        momentum, eps,
        # Tensor metadata
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Blocking
        BLOCK_SIZE=32,
    )
    
    return output
