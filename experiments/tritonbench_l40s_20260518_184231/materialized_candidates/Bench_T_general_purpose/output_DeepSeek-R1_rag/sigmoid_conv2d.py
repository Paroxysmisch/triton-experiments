import torch
import triton
import triton.language as tl

def _pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2, "Value must be int or tuple of two ints"
        return x
    else:
        raise ValueError("Value must be int or tuple of two ints")

@triton.jit
def sigmoid_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, in_channels, out_channels, groups,
    iH, iW, oH, oW, kH, kW,
    stride_H, stride_W,
    padding_H, padding_W,
    dilation_H, dilation_W,
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    weight_outc_stride, weight_inc_stride, weight_h_stride, weight_w_stride,
    output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    num_blocks_h = (oH + BLOCK_H - 1) // BLOCK_H
    block_h = pid_hw // num_blocks_h
    block_w = pid_hw % num_blocks_h
    
    start_h = block_h * BLOCK_H
    start_w = block_w * BLOCK_W
    
    offset_h = tl.arange(0, BLOCK_H)
    offset_w = tl.arange(0, BLOCK_W)
    current_h = start_h + offset_h[:, None]
    current_w = start_w + offset_w[None, :]
    
    mask_h = (current_h < oH)
    mask_w = (current_w < oW)
    mask = mask_h & mask_w
    
    in_channels_per_group = in_channels // groups
    group_id = pid_oc // (out_channels // groups)
    start_ic = group_id * in_channels_per_group
    
    accumulator = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    for ic in range(start_ic, start_ic + in_channels_per_group):
        for ky in range(kH):
            for kx in range(kW):
                iy = (current_h * stride_H) + (ky * dilation_H) - padding_H
                ix = (current_w * stride_W) + (kx * dilation_W) - padding_W
                
                iy_valid = (iy >= 0) & (iy < iH)
                ix_valid = (ix >= 0) & (ix < iW)
                valid = iy_valid & ix_valid & mask
                
                input_offset = (pid_batch * input_batch_stride) + \
                               (ic * input_channel_stride) + \
                               (iy * input_h_stride) + \
                               (ix * input_w_stride)
                input_val = tl.load(input_ptr + input_offset, mask=valid, other=0.0)
                
                weight_offset = (pid_oc * weight_outc_stride) + \
                                ((ic - start_ic) * weight_inc_stride) + \
                                (ky * weight_h_stride) + \
                                (kx * weight_w_stride)
                weight_val = tl.load(weight_ptr + weight_offset)
                
                accumulator += input_val * weight_val
    
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_oc)
        accumulator += bias_val
    
    output = 1.0 / (1.0 + tl.exp(-accumulator))
    
    output_offset = (pid_batch * output_batch_stride) + \
                    (pid_oc * output_channel_stride) + \
                    (current_h * output_h_stride) + \
                    (current_w * output_w_stride)
    
    tl.store(output_ptr + output_offset, output, mask=mask)

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    assert input.is_cuda and weight.is_cuda, "Input and weight must be CUDA tensors"
    if bias is not None:
        assert bias.is_cuda, "Bias must be a CUDA tensor"
    
    B, in_channels, iH, iW = input.shape
    out_channels, in_channels_per_group, kH, kW = weight.shape
    assert in_channels == groups * in_channels_per_group, "in_channels must be divisible by groups"
    
    stride_H, stride_W = _pair(stride)
    dilation_H, dilation_W = _pair(dilation)
    
    if isinstance(padding, str):
        if padding.lower() == 'valid':
            padding_H, padding_W = 0, 0
        elif padding.lower() == 'same':
            padH_total = max((iH - 1) * stride_H + (kH - 1) * dilation_H + 1 - iH, 0)
            padW_total = max((iW - 1) * stride_W + (kW - 1) * dilation_W + 1 - iW, 0)
            padding_H = padH_total // 2
            padding_W = padW_total // 2
        else:
            raise ValueError("Padding must be 'valid', 'same', or a tuple of integers")
    else:
        padding_H, padding_W = _pair(padding)
    
    oH = (iH + 2 * padding_H - dilation_H * (kH - 1) - 1) // stride_H + 1
    oW = (iW + 2 * padding_W - dilation_W * (kW - 1) - 1) // stride_W + 1
    assert oH > 0 and oW > 0, "Output dimensions must be positive"
    
    if out is None:
        out = torch.empty((B, out_channels, oH, oW), device=input.device, dtype=input.dtype)
    else:
        assert out.shape == (B, out_channels, oH, oW), "Output tensor has incorrect shape"
    
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride = input.stride()
    weight_outc_stride, weight_inc_stride, weight_h_stride, weight_w_stride = weight.stride()
    
    assert out_channels % groups == 0, "out_channels must be divisible by groups"
    
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    BLOCK_H, BLOCK_W = 16, 16
    num_blocks_h = (oH + BLOCK_H - 1) // BLOCK_H
    num_blocks_w = (oW + BLOCK_W - 1) // BLOCK_W
    num_blocks_hw = num_blocks_h * num_blocks_w
    
    grid = (B, out_channels, num_blocks_hw)
    
    sigmoid_conv2d_kernel[grid](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias_ptr,
        output_ptr=out.data_ptr(),
        B=B,
        in_channels=in_channels,
        out_channels=out_channels,
        groups=groups,
        iH=iH, iW=iW,
        oH=oH, oW=oW,
        kH=kH, kW=kW,
        stride_H=stride_H, stride_W=stride_W,
        padding_H=padding_H, padding_W=padding_W,
        dilation_H=dilation_H, dilation_W=dilation_W,
        input_batch_stride=input_batch_stride,
        input_channel_stride=input_channel_stride,
        input_h_stride=input_h_stride,
        input_w_stride=input_w_stride,
        weight_outc_stride=weight_outc_stride,
        weight_inc_stride=weight_inc_stride,
        weight_h_stride=weight_h_stride,
        weight_w_stride=weight_w_stride,
        output_batch_stride=out.stride(0),
        output_channel_stride=out.stride(1),
        output_h_stride=out.stride(2),
        output_w_stride=out.stride(3),
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return out
