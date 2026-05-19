import triton
import triton.language as tl

@triton.jit
def gelu_conv2d_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, 
                        stride_h, stride_w, pad_h, pad_w, 
                        dilation_h, dilation_w, 
                        in_channels, out_channels, 
                        kernel_h, kernel_w, 
                        n_elements):
    # Get the index of the current output element
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    h_idx = tl.arange(0, n_elements)

    # Calculate the input coordinates
    iH = h_idx // (input_width)
    iW = h_idx % (input_width)

    # Apply padding
    iH_padded = iH + pad_h
    iW_padded = iW + pad_w

    # Convolution operation
    conv_sum = tl.zeros((out_channels,), dtype=tl.float32)
    for kH in range(kernel_h):
        for kW in range(kernel_w):
            h_in = iH_padded + kH * dilation_h
            w_in = iW_padded + kW * dilation_w
            if 0 <= h_in < input_height and 0 <= w_in < input_width:
                conv_sum += input_ptr[batch_idx, :, h_in, w_in] * weight_ptr[out_channel_idx, :, kH, kW]

    # Add bias if provided
    if bias_ptr is not None:
        conv_sum += bias_ptr[out_channel_idx]

    # GELU activation
    output_ptr[batch_idx, out_channel_idx, iH, iW] = gelu(conv_sum)

@triton.jit
def gelu(x):
    return x * tl.sigmoid(0.5 * (1 + tl.tanh(tl.sqrt(2 / tl.pi) * (x + 0.044715 * x**3))))

def gelu_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, 
                 stride: Union[int, Tuple[int, int]] = 1, 
                 padding: Union[int, Tuple[int, int], str] = 0, 
                 dilation: Union[int, Tuple[int, int]] = 1, 
                 groups: int = 1, 
                 approximate: str = 'none', 
                 out: Optional[Tensor] = None) -> Tensor:
    # ... existing code ...
    
    # Prepare parameters for the kernel
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation

    # Call the Triton kernel
    gelu_conv2d_kernel[(grid_size,)](input_ptr, weight_ptr, bias_ptr, output_ptr, 
                                      stride_h, stride_w, pad_h, pad_w, 
                                      dilation_h, dilation_w, 
                                      in_channels, out_channels, 
                                      kernel_h, kernel_w, 
                                      n_elements)

    return output_tensor
    # ... existing code ...
