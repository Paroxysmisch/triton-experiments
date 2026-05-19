import torch
import triton
import triton.language as tl

@triton.jit
def _fused_silu_ln_conv2d_kernel(
    # Pointers to matrices
    x_ptr, weight_ptr, conv_weight_ptr, conv_bias_ptr,
    output_ptr,
    # Matrix dimensions
    batch_size, in_channels, height, width,
    out_channels, kernel_size,
    # Convolution parameters
    stride, padding, dilation, groups,
    # Layer norm parameter
    eps,
    # Strides for the different tensors
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_o, w_stride_i, w_stride_h, w_stride_w,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_SIZE: tl.constexpr
):
    # Compute indices
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Calculate batch and channel indices
    batch_idx = pid // (out_channels * out_height * out_width)
    tmp = pid % (out_channels * out_height * out_width)
    channel_idx = tmp // (out_height * out_width)
    h_idx = (tmp // out_width) % out_height
    w_idx = tmp % out_width

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Convolution
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            for ic in range(in_channels):
                h_in = h_idx * stride - padding + kh * dilation
                w_in = w_idx * stride - padding + kw * dilation
                
                if 0 <= h_in < height and 0 <= w_in < width:
                    x_idx = batch_idx * x_stride_b + ic * x_stride_c + h_in * x_stride_h + w_in * x_stride_w
                    w_idx = channel_idx * w_stride_o + ic * w_stride_i + kh * w_stride_h + kw * w_stride_w
                    
                    x_val = tl.load(x_ptr + x_idx)
                    w_val = tl.load(conv_weight_ptr + w_idx)
                    acc += x_val * w_val

    # Add bias if present
    if conv_bias_ptr is not None:
        acc += tl.load(conv_bias_ptr + channel_idx)

    # Layer Normalization
    mean = tl.sum(acc, axis=0) / BLOCK_SIZE
    var = tl.sum((acc - mean) ** 2, axis=0) / BLOCK_SIZE
    inv_std = 1 / tl.sqrt(var + eps)
    normalized = (acc - mean) * inv_std
    
    # Scale with learnable weight
    weight = tl.load(weight_ptr + channel_idx)
    normalized = normalized * weight

    # SiLU activation
    output = normalized * tl.sigmoid(normalized)
    
    # Store result
    out_idx = (batch_idx * out_stride_b + channel_idx * out_stride_c + 
               h_idx * out_stride_h + w_idx * out_stride_w)
    tl.store(output_ptr + out_idx, output)

def fused_silu_layer_norm_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor = None,
    conv_stride: int = 1,
    conv_padding: int = 0,
    conv_dilation: int = 1,
    conv_groups: int = 1,
    ln_eps: float = 1e-5
) -> torch.Tensor:
    """
    Fused 2D Convolution + Layer Normalization + SiLU activation.
    
    Args:
        x (Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        weight (Tensor): Layer norm weights of shape (out_channels,)
        conv_weight (Tensor): Convolution weights
        conv_bias (Tensor, optional): Convolution bias
        conv_stride (int): Convolution stride. Default: 1
        conv_padding (int): Convolution padding. Default: 0
        conv_dilation (int): Convolution dilation. Default: 1
        conv_groups (int): Convolution groups. Default: 1
        ln_eps (float): Layer normalization epsilon. Default: 1e-5
    
    Returns:
        Tensor: Output tensor after convolution, normalization and activation
    """
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = conv_weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Prepare output tensor
    output = torch.empty(
        (batch_size, out_channels, out_height, out_width),
        device=x.device,
        dtype=x.dtype
    )
    
    # Define grid and block sizes
    BLOCK_SIZE = 32
    grid = (batch_size * out_channels * out_height * out_width,)
    
    # Launch kernel
    _fused_silu_ln_conv2d_kernel[grid](
        x, weight, conv_weight, 
        conv_bias if conv_bias is not None else x.new_zeros(0),
        output,
        batch_size, in_channels, height, width,
        out_channels, kernel_size,
        conv_stride, conv_padding, conv_dilation, conv_groups,
        ln_eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2), conv_weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE
    )
    
    return output
