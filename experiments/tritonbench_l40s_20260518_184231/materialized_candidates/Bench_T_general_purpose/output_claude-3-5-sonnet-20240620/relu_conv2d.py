import torch
import triton
import triton.language as tl

@triton.jit
def relu_conv2d_kernel(
    # Pointers to matrices
    output_ptr, input_ptr, weight_ptr, bias_ptr,
    # Matrix dimensions
    batch, in_channels, out_channels, in_height, in_width,
    kernel_height, kernel_width,
    out_height, out_width,
    # Conv parameters
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    # Other parameters
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    # Position of elements
    pid = tl.program_id(0)
    
    # Calculate output position
    n = pid // (out_channels * out_height * out_width)
    rem = pid % (out_channels * out_height * out_width)
    oc = rem // (out_height * out_width)
    h_out = (rem // out_width) % out_height
    w_out = rem % out_width
    
    # Calculate input bounds
    channels_per_group = in_channels // groups
    group_id = oc // (out_channels // groups)
    
    # Initialize accumulator
    acc = 0.0
    
    # Convolution loop
    for ic in range(channels_per_group):
        ic_idx = group_id * channels_per_group + ic
        for kh in range(kernel_height):
            h_in = h_out * stride_h - padding_h + kh * dilation_h
            if not (0 <= h_in < in_height):
                continue
                
            for kw in range(kernel_width):
                w_in = w_out * stride_w - padding_w + kw * dilation_w
                if not (0 <= w_in < in_width):
                    continue
                    
                # Get input value
                in_idx = (n * in_channels * in_height * in_width +
                         ic_idx * in_height * in_width +
                         h_in * in_width + w_in)
                input_val = tl.load(input_ptr + in_idx)
                
                # Get weight value
                weight_idx = (oc * channels_per_group * kernel_height * kernel_width +
                            ic * kernel_height * kernel_width +
                            kh * kernel_width + kw)
                weight_val = tl.load(weight_ptr + weight_idx)
                
                acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + oc)
    
    # Apply ReLU
    acc = tl.maximum(acc, 0.0)
    
    # Store output
    out_idx = (n * out_channels * out_height * out_width +
               oc * out_height * out_width +
               h_out * out_width + w_out)
    tl.store(output_ptr + out_idx, acc)

def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    # Input validation
    assert input.dim() == 4, "Input must be 4D tensor"
    assert weight.dim() == 4, "Weight must be 4D tensor"
    
    # Handle scalar and tuple parameters
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
        
    batch, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    out_height = ((in_height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0]) + 1
    out_width = ((in_width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1]) + 1
    
    # Create output tensor
    output = torch.empty((batch, out_channels, out_height, out_width), 
                        device=input.device, dtype=input.dtype)
    
    # Launch kernel
    grid = (batch * out_channels * out_height * out_width,)
    relu_conv2d_kernel[grid](
        output, input, weight,
        bias if bias is not None else None,
        batch, in_channels, out_channels,
        in_height, in_width,
        kernel_height, kernel_width,
        out_height, out_width,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups,
        BLOCK_SIZE=32,
    )
    
    return output
