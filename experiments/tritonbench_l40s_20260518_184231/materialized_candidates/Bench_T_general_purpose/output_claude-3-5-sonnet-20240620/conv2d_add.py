import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(input_ptr, weight_ptr, bias_ptr, other_ptr, out_ptr, 
                  stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, 
                  groups, alpha, n_batch, in_channels, out_channels, 
                  iH, iW, kH, kW):
    # Calculate the output dimensions
    oH = (iH + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1

    # Iterate over the output dimensions
    batch = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)

    # Initialize output
    out_val = 0.0

    # Convolution operation
    for c in range(in_channels // groups):
        for kh in range(kH):
            for kw in range(kW):
                ih = oh * stride_h - pad_h + kh * dilation_h
                iw = ow * stride_w - pad_w + kw * dilation_w
                if 0 <= ih < iH and 0 <= iw < iW:
                    input_val = tl.load(input_ptr[batch, c, ih, iw])
                    weight_val = tl.load(weight_ptr[0, c, kh, kw])
                    out_val += input_val * weight_val

    # Add bias if provided
    if bias_ptr is not None:
        out_val += tl.load(bias_ptr)

    # Add other tensor or scalar
    if other_ptr is not None:
        other_val = tl.load(other_ptr) if isinstance(other_ptr, tl.Tensor) else other_ptr
        out_val += alpha * other_val

    # Store the result
    tl.store(out_ptr[batch, 0, oh, ow], out_val)

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, 
                dilation=1, groups=1, alpha=1, out=None):
    # Get input dimensions
    n_batch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Handle padding
    if isinstance(padding, str):
        if padding == 'valid':
            pad_h, pad_w = 0, 0
        elif padding == 'same':
            pad_h = (kH - 1) // 2
            pad_w = (kW - 1) // 2
    else:
        pad_h, pad_w = padding, padding

    # Launch the kernel
    grid = (n_batch, (iH + stride - 1) // stride, (iW + stride - 1) // stride)
    conv2d_kernel[grid](input, weight, bias, other, out, 
                        stride_h=stride, stride_w=stride, 
                        pad_h=pad_h, pad_w=pad_w, 
                        dilation_h=dilation, dilation_w=dilation, 
                        groups=groups, alpha=alpha, 
                        n_batch=n_batch, in_channels=in_channels, 
                        out_channels=out_channels, 
                        iH=iH, iW=iW, kH=kH, kW=kW)
