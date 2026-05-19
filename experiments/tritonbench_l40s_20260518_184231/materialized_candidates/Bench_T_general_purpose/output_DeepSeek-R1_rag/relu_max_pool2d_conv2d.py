import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Convolution parameters
    in_channels, out_channels, 
    iH, iW, kH, kW,
    conv_stride_h, conv_stride_w,
    conv_padding_h, conv_padding_w,
    conv_dilation_h, conv_dilation_w,
    groups,
    # Output dimensions
    oH, oW,
    # Blocking
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)

    # Offsets for spatial dimensions
    oh_offsets = pid_oh * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    ow_offsets = pid_ow * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    oc_offsets = pid_channel * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    # Masks for valid positions
    oh_mask = oh_offsets < oH
    ow_mask = ow_offsets < oW
    oc_mask = oc_offsets < out_channels

    # Group handling
    group_id = oc_offsets // (out_channels // groups)
    ic_start = group_id * (in_channels // groups)
    ic_end = (group_id + 1) * (in_channels // groups)

    accumulator = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    for kh in range(kH):
        for kw in range(kW):
            for ic in range(ic_start, ic_end):
                # Calculate input positions with dilation and padding
                ih = (oh_offsets[:, None, None] * conv_stride_h) + (kh * conv_dilation_h) - conv_padding_h
                iw = (ow_offsets[None, :, None] * conv_stride_w) + (kw * conv_dilation_w) - conv_padding_w

                # Check boundaries
                ih_mask = (ih >= 0) & (ih < iH) & oh_mask[:, None, None]
                iw_mask = (iw >= 0) & (iw < iW) & ow_mask[None, :, None]
                valid = ih_mask & iw_mask

                input_val = tl.load(
                    input_ptr + pid_batch * in_channels * iH * iW + 
                    ic * iH * iW + ih * iW + iw,
                    mask=valid,
                    other=0.0
                )

                weight_val = tl.load(
                    weight_ptr + oc_offsets * (in_channels // groups) * kH * kW + 
                    (ic - ic_start) * kH * kW + kh * kW + kw,
                    mask=oc_mask[:, None, None],
                    other=0.0
                )

                accumulator += input_val * weight_val[:, None, None]

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
        accumulator += bias[:, None, None]

    # Store output
    output_offset = (
        pid_batch * out_channels * oH * oW +
        oc_offsets[:, None, None] * oH * oW +
        oh_offsets[:, None, None] * oW +
        ow_offsets[None, :, None]
    )
    tl.store(output_ptr + output_offset, accumulator, mask=oc_mask[:, None, None] & oh_mask[:, None, None] & ow_mask[None, :, None])

@triton.jit
def max_pool2d_kernel(
    input_ptr, output_ptr,
    # Pool parameters
    pool_kH, pool_kW,
    pool_stride_h, pool_stride_w,
    pool_padding_h, pool_padding_w,
    # Input dimensions
    iH, iW,
    # Output dimensions
    oH, oW,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)

    oh_start = pid_oh * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    ow_start = pid_ow * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    oh_mask = oh_start < oH
    ow_mask = ow_start < oW

    max_values = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32) - float('inf')

    for kh in range(pool_kH):
        for kw in range(pool_kW):
            ih = oh_start[:, None] * pool_stride_h - pool_padding_h + kh
            iw = ow_start[None, :] * pool_stride_w - pool_padding_w + kw

            in_bounds = (ih >= 0) & (ih < iH) & (iw >= 0) & (iw < iW) & oh_mask[:, None] & ow_mask[None, :]

            input_val = tl.load(
                input_ptr + pid_batch * iH * iW + pid_channel * iH * iW * pid_batch + 
                ih * iW + iw,
                mask=in_bounds,
                other=-float('inf')
            )

            max_values = tl.maximum(max_values, input_val)

    output_offset = (
        pid_batch * oH * oW +
        pid_channel * oH * oW * pid_batch +
        oh_start[:, None] * oW +
        ow_start[None, :]
    )
    tl.store(output_ptr + output_offset, max_values, mask=oh_mask[:, None] & ow_mask[None, :])

@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def relu_max_pool2d_conv2d(
    input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1,
    pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False
):
    assert input.is_cuda and weight.is_cuda, "Input and weight must be on CUDA device"
    if bias is not None:
        assert bias.is_cuda, "Bias must be on CUDA device"
    
    # Compute convolution output shape
    def compute_output_size(input_size, kernel_size, stride, padding, dilation):
        return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    iH, iW = input.shape[2], input.shape[3]
    kH, kW = weight.shape[2], weight.shape[3]
    conv_stride = (conv_stride, conv_stride) if isinstance(conv_stride, int) else conv_stride
    conv_padding = (conv_padding, conv_padding) if isinstance(conv_padding, int) else conv_padding
    conv_dilation = (conv_dilation, conv_dilation) if isinstance(conv_dilation, int) else conv_dilation
    
    oH = compute_output_size(iH, kH, conv_stride[0], conv_padding[0], conv_dilation[0])
    oW = compute_output_size(iW, kW, conv_stride[1], conv_padding[1], conv_dilation[1])
    
    # Allocate convolution output
    batch, in_channels = input.shape[0], input.shape[1]
    out_channels = weight.shape[0]
    conv_output = torch.empty((batch, out_channels, oH, oW), device=input.device)
    
    # Launch convolution kernel
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 4
    grid_conv = (
        batch,
        triton.cdiv(out_channels, BLOCK_SIZE_C),
        triton.cdiv(oH, BLOCK_SIZE_H),
        triton.cdiv(oW, BLOCK_SIZE_W),
    )
    conv2d_kernel[grid_conv](
        input, weight, bias, conv_output,
        in_channels, out_channels, iH, iW, kH, kW,
        conv_stride[0], conv_stride[1],
        conv_padding[0], conv_padding[1],
        conv_dilation[0], conv_dilation[1],
        conv_groups,
        oH, oW,
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C,
    )
    
    # Compute max pooling output shape
    pool_kernel_size = (pool_kernel_size, pool_kernel_size) if isinstance(pool_kernel_size, int) else pool_kernel_size
    pool_stride = pool_stride if pool_stride is not None else pool_kernel_size
    pool_stride = (pool_stride, pool_stride) if isinstance(pool_stride, int) else pool_stride
    pool_padding = (pool_padding, pool_padding) if isinstance(pool_padding, int) else pool_padding
    pool_dilation = (pool_dilation, pool_dilation) if isinstance(pool_dilation, int) else pool_dilation
    
    def pool_output_size(input_size, kernel_size, stride, padding, dilation, ceil_mode):
        numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
        if ceil_mode:
            return (numerator + stride - 1) // stride + 1
        else:
            return numerator // stride + 1
    
    pool_H = pool_output_size(oH, pool_kernel_size[0], pool_stride[0], pool_padding[0], pool_dilation[0], pool_ceil_mode)
    pool_W = pool_output_size(oW, pool_kernel_size[1], pool_stride[1], pool_padding[1], pool_dilation[1], pool_ceil_mode)
    
    # Allocate max pool output
    pool_output = torch.empty((batch, out_channels, pool_H, pool_W), device=input.device)
    
    # Launch max pool kernel
    BLOCK_SIZE_PH = 16
    BLOCK_SIZE_PW = 16
    grid_pool = (
        batch,
        out_channels,
        triton.cdiv(pool_H, BLOCK_SIZE_PH),
        triton.cdiv(pool_W, BLOCK_SIZE_PW),
    )
    max_pool2d_kernel[grid_pool](
        conv_output, pool_output,
        pool_kernel_size[0], pool_kernel_size[1],
        pool_stride[0], pool_stride[1],
        pool_padding[0], pool_padding[1],
        oH, oW,
        pool_H, pool_W,
        BLOCK_SIZE_PH, BLOCK_SIZE_PW,
    )
    
    # Apply ReLU
    if inplace:
        output = pool_output
    else:
        output = torch.empty_like(pool_output)
    
    n_elements = pool_output.numel()
    grid_relu = (triton.cdiv(n_elements, 1024),)
    relu_kernel[grid_relu](
        pool_output.data_ptr(), output.data_ptr(), n_elements, BLOCK_SIZE=1024
    )
    
    return output
