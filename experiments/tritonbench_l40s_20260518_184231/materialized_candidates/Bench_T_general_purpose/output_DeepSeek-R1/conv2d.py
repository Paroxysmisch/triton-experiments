import torch
import triton
import triton.language as tl

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Tensor dimensions
    n, c_in, h_in, w_in,
    c_out, k_h, k_w,
    # Kernel parameters
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    # Tensor strides
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    weight_outc_stride, weight_inc_stride, weight_h_stride, weight_w_stride,
    output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
    # Meta parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output indices
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Generate offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Compute output position
    oh = pid_h * stride_h - pad_h
    ow = pid_w * stride_w - pad_w
    
    # Accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels (grouped)
    group_size = c_in // groups
    group_id = pid_channel // (c_out // groups)
    c_start = group_id * group_size
    c_end = (group_id + 1) * group_size
    
    for c in range(c_start, c_end):
        # Loop over kernel height and width
        for kh in range(k_h):
            for kw in range(k_w):
                # Calculate input position with dilation
                ih = oh + kh * dilation_h
                iw = ow + kw * dilation_w
                # Check if input position is within bounds
                if ih >= 0 and ih < h_in and iw >= 0 and iw < w_in:
                    # Load input and weight
                    input_idx = pid_batch * input_batch_stride + c * input_channel_stride + ih * input_h_stride + iw * input_w_stride
                    input_val = tl.load(input_ptr + input_idx + offsets, mask=(offsets < n), other=0.0)
                    weight_idx = pid_channel * weight_outc_stride + c * weight_inc_stride + kh * weight_h_stride + kw * weight_w_stride
                    weight_val = tl.load(weight_ptr + weight_idx + offsets, mask=(offsets < (c_end - c_start)), other=0.0)
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_channel + offsets, mask=(offsets < c_out), other=0.0)
        acc += bias_val
    
    # Compute output index
    output_idx = pid_batch * output_batch_stride + pid_channel * output_channel_stride + pid_h * output_h_stride + pid_w * output_w_stride
    tl.store(output_ptr + output_idx + offsets, acc, mask=(offsets < n))

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Process stride, padding, dilation
    def expand(x, name):
        if isinstance(x, int):
            return (x, x)
        elif isinstance(x, tuple) and len(x) == 2:
            return x
        else:
            raise ValueError(f"{name} must be an int or a tuple of two ints")
    
    stride = expand(stride, "stride")
    dilation = expand(dilation, "dilation")
    
    if isinstance(padding, str):
        if padding.lower() == 'valid':
            padding = (0, 0)
        elif padding.lower() == 'same':
            k_h, k_w = weight.shape[2], weight.shape[3]
            pad_h = ((input.shape[2] - 1) * stride[0] + (k_h - 1) * dilation[0] + 1 - input.shape[2]) // 2
            pad_w = ((input.shape[3] - 1) * stride[1] + (k_w - 1) * dilation[1] + 1 - input.shape[3]) // 2
            padding = (pad_h, pad_w)
        else:
            raise ValueError("padding must be 'valid', 'same', or a tuple of ints")
    else:
        padding = expand(padding, "padding")
    
    # Compute output shape
    n, c_in, h_in, w_in = input.shape
    c_out, c_in_g, k_h, k_w = weight.shape
    assert c_in == c_in_g * groups, "in_channels must be divisible by groups"
    
    h_out = (h_in + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    w_out = (w_in + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    
    # Allocate output tensor
    output = torch.empty((n, c_out, h_out, w_out), dtype=input.dtype, device=input.device)
    
    # Grid and block dimensions
    grid = (n, c_out, h_out, w_out)
    BLOCK_SIZE = 1  # Simplified for demonstration; adjust based on hardware
    
    # Launch kernel
    _conv2d_kernel[grid](
        input, weight, bias, output,
        n, c_in, h_in, w_in,
        c_out, k_h, k_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
