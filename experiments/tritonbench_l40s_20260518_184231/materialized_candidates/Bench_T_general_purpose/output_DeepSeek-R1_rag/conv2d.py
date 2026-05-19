import torch
import triton
import triton.language as tl

def _pair(x):
    return (x, x) if isinstance(x, int) else x

def _same_padding(input_size, kernel_size, stride, dilation):
    effective_kernel = dilation * (kernel_size - 1) + 1
    return max(0, (input_size * (stride - 1) + effective_kernel - stride) // 2)

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    groups,
    in_channels,
    out_channels,
    iH, iW,
    kH, kW,
    oH, oW,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
):
    # Program IDs
    batch = tl.program_id(0)
    oc = tl.program_id(1)
    y_block = tl.program_id(2)
    x_block = tl.program_id(3)

    # Calculate output positions
    y_start = y_block * BLOCK_SIZE_H
    y_positions = y_start + tl.arange(0, BLOCK_SIZE_H)
    y_mask = y_positions < oH

    x_start = x_block * BLOCK_SIZE_W
    x_positions = x_start + tl.arange(0, BLOCK_SIZE_W)
    x_mask = x_positions < oW

    mask = y_mask[:, None] & x_mask[None, :]

    # Group calculations
    group = oc // (out_channels // groups)
    in_channels_per_group = in_channels // groups
    oc_in_group = oc % (out_channels // groups)

    # Initialize accumulators
    if IS_COMPLEX:
        acc_real = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
        acc_imag = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Kernel loops
    for kh in range(kH):
        for kw in range(kW):
            iy = y_positions * stride_h + kh * dilation_h - padding_h
            ix = x_positions * stride_w + kw * dilation_w - padding_w

            iy_valid = (iy >= 0) & (iy < iH)
            ix_valid = (ix >= 0) & (ix < iW)
            valid = iy_valid[:, None] & ix_valid[None, :]

            for ic in range(in_channels_per_group):
                input_ic = group * in_channels_per_group + ic
                elem_size = 2 if IS_COMPLEX else 1

                # Input offset calculation
                input_offset = (
                    batch * in_channels * iH * iW * elem_size +
                    input_ic * iH * iW * elem_size +
                    (iy * iW + ix) * elem_size
                )

                # Weight offset calculation
                weight_offset = (
                    oc * (in_channels_per_group * kH * kW) * elem_size +
                    (ic * kH * kW + kh * kW + kw) * elem_size
                )

                if IS_COMPLEX:
                    # Complex operations
                    in_real = tl.load(input_ptr + input_offset, mask=valid, other=0.0)
                    in_imag = tl.load(input_ptr + input_offset + 1, mask=valid, other=0.0)
                    w_real = tl.load(weight_ptr + weight_offset)
                    w_imag = tl.load(weight_ptr + weight_offset + 1)
                    
                    acc_real += in_real * w_real - in_imag * w_imag
                    acc_imag += in_real * w_imag + in_imag * w_real
                else:
                    # Real operations
                    input_val = tl.load(input_ptr + input_offset, mask=valid, other=0.0)
                    weight_val = tl.load(weight_ptr + weight_offset)
                    accumulator += input_val * weight_val

    # Add bias
    if bias_ptr is not None:
        if IS_COMPLEX:
            bias_real = tl.load(bias_ptr + oc * 2)
            bias_imag = tl.load(bias_ptr + oc * 2 + 1)
            acc_real += bias_real
            acc_imag += bias_imag
        else:
            bias_val = tl.load(bias_ptr + oc)
            accumulator += bias_val

    # Store results
    elem_size = 2 if IS_COMPLEX else 1
    output_offset = (
        batch * out_channels * oH * oW * elem_size +
        oc * oH * oW * elem_size +
        (y_positions[:, None] * oW + x_positions[None, :]) * elem_size
    )

    if IS_COMPLEX:
        tl.store(output_ptr + output_offset, acc_real, mask=mask)
        tl.store(output_ptr + output_offset + 1, acc_imag, mask=mask)
    else:
        tl.store(output_ptr + output_offset, accumulator, mask=mask)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Device checks
    assert input.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    if bias is not None: 
        assert bias.is_cuda, "Bias must be on CUDA"

    # Shape validation
    B, C_in, iH, iW = input.shape
    C_out, C_in_g, kH, kW = weight.shape
    assert C_in % groups == 0, "in_channels must be divisible by groups"
    assert C_out % groups == 0, "out_channels must be divisible by groups"
    assert C_in_g == C_in // groups, "Weight shape mismatch"

    # Parameter expansion
    stride_h, stride_w = _pair(stride)
    dilation_h, dilation_w = _pair(dilation)
    
    # Padding handling
    if isinstance(padding, str):
        if padding == 'same':
            pad_h = _same_padding(iH, kH, stride_h, dilation_h)
            pad_w = _same_padding(iW, kW, stride_w, dilation_w)
        else:  # valid
            pad_h, pad_w = 0, 0
    else:
        pad_h, pad_w = _pair(padding)

    # Output dimensions
    oH = (iH + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1
    assert oH > 0 and oW > 0, "Invalid output dimensions"

    # Contiguous tensors
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Output allocation
    output = torch.empty(
        (B, C_out, oH, oW), 
        device=input.device,
        dtype=input.dtype
    )

    # Kernel configuration
    BLOCK_H, BLOCK_W = 16, 16
    grid_batch = B
    grid_oc = C_out
    grid_y = (oH + BLOCK_H - 1) // BLOCK_H
    grid_x = (oW + BLOCK_W - 1) // BLOCK_W

    # Determine complex mode
    is_complex = input.is_complex()
    if is_complex:
        assert all(t.is_complex() for t in [input, weight, *([bias] if bias else [])]), "Mixed complex/real tensors"

    # Launch kernel
    conv2d_kernel[(grid_batch, grid_oc, grid_y, grid_x)](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        bias_ptr=bias if bias is not None else None,
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=pad_h,
        padding_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        groups=groups,
        in_channels=C_in,
        out_channels=C_out,
        iH=iH, iW=iW,
        kH=kH, kW=kW,
        oH=oH, oW=oW,
        BLOCK_SIZE_H=BLOCK_H,
        BLOCK_SIZE_W=BLOCK_W,
        IS_COMPLEX=is_complex,
    )

    return output
