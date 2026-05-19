import torch
import triton
import triton.language as tl

@triton.jit
def relu_max_pool2d_conv2d_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    # Matrix dimensions
    batch, in_channels, out_channels, in_height, in_width,
    out_height, out_width, kernel_h, kernel_w,
    # Convolution parameters
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
    # Pooling parameters
    pool_size_h, pool_size_w, pool_stride_h, pool_stride_w,
    pool_padding_h, pool_padding_w, pool_dilation_h, pool_dilation_w,
    # Other parameters
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_o, weight_stride_i, weight_stride_h, weight_stride_w,
    output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE: tl.constexpr):
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate position
    batch_idx = pid // (out_channels * out_height * out_width)
    tmp = pid % (out_channels * out_height * out_width)
    oc = tmp // (out_height * out_width)
    tmp = tmp % (out_height * out_width)
    oh = tmp // out_width
    ow = tmp % out_width

    # Initialize accumulator
    acc = 0.0
    if bias_ptr:
        acc = tl.load(bias_ptr + oc)

    # Convolution
    for ic in range(in_channels // groups):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                ih = oh * stride_h - padding_h + kh * dilation_h
                iw = ow * stride_w - padding_w + kw * dilation_w
                
                if 0 <= ih < in_height and 0 <= iw < in_width:
                    input_idx = (batch_idx * input_stride_b + 
                               ic * input_stride_c +
                               ih * input_stride_h +
                               iw * input_stride_w)
                    weight_idx = (oc * weight_stride_o +
                                ic * weight_stride_i +
                                kh * weight_stride_h +
                                kw * weight_stride_w)
                    
                    inp_val = tl.load(input_ptr + input_idx)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    acc += inp_val * weight_val

    # Max pooling
    pool_result = float('-inf')
    for ph in range(pool_size_h):
        for pw in range(pool_size_w):
            h_idx = oh * pool_stride_h + ph * pool_dilation_h - pool_padding_h
            w_idx = ow * pool_stride_w + pw * pool_dilation_w - pool_padding_w
            
            if 0 <= h_idx < out_height and 0 <= w_idx < out_width:
                val = acc
                pool_result = tl.maximum(pool_result, val)

    # ReLU activation
    output = tl.maximum(pool_result, 0.0)
    
    # Store result
    output_idx = (batch_idx * output_stride_b +
                 oc * output_stride_c +
                 oh * output_stride_h +
                 ow * output_stride_w)
    tl.store(output_ptr + output_idx, output)

def relu_max_pool2d_conv2d(input, weight, bias=None, 
                          conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1,
                          pool_kernel_size=2, pool_stride=None, pool_padding=0,
                          pool_dilation=1, pool_ceil_mode=False, inplace=False):
    
    # Input validation and shape calculations
    assert input.dim() == 4, "Input must be 4D tensor"
    assert weight.dim() == 4, "Weight must be 4D tensor"
    
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Handle stride parameters
    if isinstance(conv_stride, int):
        conv_stride = (conv_stride, conv_stride)
    if isinstance(conv_padding, int):
        conv_padding = (conv_padding, conv_padding)
    if isinstance(conv_dilation, int):
        conv_dilation = (conv_dilation, conv_dilation)
        
    # Calculate output dimensions after convolution
    out_height = ((in_height + 2 * conv_padding[0] - conv_dilation[0] * (kernel_h - 1) - 1) // conv_stride[0] + 1)
    out_width = ((in_width + 2 * conv_padding[1] - conv_dilation[1] * (kernel_w - 1) - 1) // conv_stride[1] + 1)
    
    # Handle pooling parameters
    if isinstance(pool_kernel_size, int):
        pool_kernel_size = (pool_kernel_size, pool_kernel_size)
    if pool_stride is None:
        pool_stride = pool_kernel_size
    elif isinstance(pool_stride, int):
        pool_stride = (pool_stride, pool_stride)
    if isinstance(pool_padding, int):
        pool_padding = (pool_padding, pool_padding)
    if isinstance(pool_dilation, int):
        pool_dilation = (pool_dilation, pool_dilation)
    
    # Calculate final output dimensions
    final_height = ((out_height + 2 * pool_padding[0] - pool_dilation[0] * (pool_kernel_size[0] - 1) - 1) // pool_stride[0] + 1)
    final_width = ((out_width + 2 * pool_padding[1] - pool_dilation[1] * (pool_kernel_size[1] - 1) - 1) // pool_stride[1] + 1)
    
    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, final_height, final_width), 
                        device=input.device, dtype=input.dtype)
    
    # Launch kernel
    grid = (batch_size * out_channels * final_height * final_width,)
    relu_max_pool2d_conv2d_kernel[grid](
        input, weight, output, bias if bias is not None else None,
        batch_size, in_channels, out_channels, in_height, in_width,
        final_height, final_width, kernel_h, kernel_w,
        conv_stride[0], conv_stride[1], conv_padding[0], conv_padding[1],
        conv_dilation[0], conv_dilation[1], conv_groups,
        pool_kernel_size[0], pool_kernel_size[1], pool_stride[0], pool_stride[1],
        pool_padding[0], pool_padding[1], pool_dilation[0], pool_dilation[1],
        *input.stride(), *weight.stride(), *output.stride(),
        BLOCK_SIZE=32
    )
    
    return output
