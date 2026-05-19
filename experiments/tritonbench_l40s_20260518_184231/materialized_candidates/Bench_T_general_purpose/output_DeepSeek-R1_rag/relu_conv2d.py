import torch
import triton
import triton.language as tl

def _pair(x):
    if isinstance(x, (list, tuple)):
        if len(x) != 2:
            raise ValueError(f"Padding must be a tuple/list of two integers, got {x}")
        return (int(x[0]), int(x[1]))
    return (int(x), int(x))

def _calculate_same_padding(input_size, kernel_size, stride, dilation):
    output_size = (input_size + stride - 1) // stride
    total_padding = max(0, (output_size - 1) * stride + (kernel_size - 1) * dilation + 1 - input_size)
    return total_padding // 2, total_padding - total_padding // 2

@triton.jit
def relu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Tensor dimensions
    batch_size, in_channels, iH, iW,
    out_channels, kH, kW,
    # Convolution parameters
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    groups,
    in_channels_per_group, out_channels_per_group,
    # Output dimensions
    oH, oW,
    # Meta parameters
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_ob = tl.program_id(2)

    # Calculate spatial block indices
    num_blocks_h = (oH + BLOCK_H - 1) // BLOCK_H
    block_h = pid_ob // ((oW + BLOCK_W - 1) // BLOCK_W)
    block_w = pid_ob % ((oW + BLOCK_W - 1) // BLOCK_W)

    oh_start = block_h * BLOCK_H
    ow_start = block_w * BLOCK_W

    # Offsets for oh and ow in this block
    ohs = oh_start + tl.arange(0, BLOCK_H)
    ows = ow_start + tl.arange(0, BLOCK_W)
    oh_mask = ohs < oH
    ow_mask = ows < oW

    group_idx = pid_oc // out_channels_per_group
    c_start = group_idx * in_channels_per_group

    accumulator = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    for kh in range(kH):
        for kw in range(kW):
            # Compute input positions
            ihs = ohs[:, None] * stride_h + kh * dilation_h - padding_h
            iws = ows[None, :] * stride_w + kw * dilation_w - padding_w

            ih_mask = (ihs >= 0) & (ihs < iH)
            iw_mask = (iws >= 0) & (iws < iW)
            valid_mask = ih_mask[:, None] & iw_mask[None, :] & oh_mask[:, None] & ow_mask[None, :]

            for c in range(in_channels_per_group):
                input_c = c_start + c
                input_idx = pid_batch * in_channels * iH * iW + input_c * iH * iW + ihs[:, None] * iW + iws[None, :]
                input_val = tl.load(input_ptr + input_idx, mask=valid_mask, other=0.0)

                weight_idx = pid_oc * in_channels_per_group * kH * kW + c * kH * kW + kh * kW + kw
                weight_val = tl.load(weight_ptr + weight_idx)

                accumulator += input_val * weight_val

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_oc)
        accumulator += bias

    accumulator = tl.maximum(accumulator, 0.0)

    output_idx = pid_batch * out_channels * oH * oW + pid_oc * oH * oW + ohs[:, None] * oW + ows[None, :]
    tl.store(output_ptr + output_idx, accumulator, mask=oh_mask[:, None] & ow_mask[None, :])

def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    assert input.is_cuda and weight.is_cuda, "Input and weight must be on CUDA device"
    assert input.dim() == 4, "Input must be 4D (N, C, H, W)"
    assert weight.dim() == 4, "Weight must be 4D (O, I, KH, KW)"
    
    in_channels = input.size(1)
    out_channels = weight.size(0)
    kH, kW = weight.shape[2], weight.shape[3]
    
    assert in_channels % groups == 0, "in_channels must be divisible by groups"
    assert out_channels % groups == 0, "out_channels must be divisible by groups"
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    stride_h, stride_w = _pair(stride)
    dilation_h, dilation_w = _pair(dilation)
    
    if isinstance(padding, str):
        if padding.lower() == 'valid':
            padding_h, padding_w = 0, 0
        elif padding.lower() == 'same':
            padding_h, _ = _calculate_same_padding(input.size(2), kH, stride_h, dilation_h)
            padding_w, _ = _calculate_same_padding(input.size(3), kW, stride_w, dilation_w)
        else:
            raise ValueError(f"Invalid padding string: {padding}")
    else:
        padding_h, padding_w = _pair(padding)
    
    iH, iW = input.shape[2], input.shape[3]
    oH = (iH + 2 * padding_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    oW = (iW + 2 * padding_w - dilation_w * (kW - 1) - 1) // stride_w + 1

    output = torch.empty((input.size(0), out_channels, oH, oW), device=input.device, dtype=input.dtype)
    
    BLOCK_H, BLOCK_W = 16, 16
    grid_batch = input.size(0)
    grid_oc = out_channels
    num_blocks_h = (oH + BLOCK_H - 1) // BLOCK_H
    num_blocks_w = (oW + BLOCK_W - 1) // BLOCK_W
    grid_ob = num_blocks_h * num_blocks_w
    
    bias_ptr = bias.data_ptr() if bias is not None else None
    relu_conv2d_kernel[(grid_batch, grid_oc, grid_ob)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias_ptr,
        output_ptr=output,
        batch_size=input.size(0),
        in_channels=in_channels,
        iH=iH,
        iW=iW,
        out_channels=out_channels,
        kH=kH,
        kW=kW,
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=padding_h,
        padding_w=padding_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        groups=groups,
        in_channels_per_group=in_channels_per_group,
        out_channels_per_group=out_channels_per_group,
        oH=oH,
        oW=oW,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return output
