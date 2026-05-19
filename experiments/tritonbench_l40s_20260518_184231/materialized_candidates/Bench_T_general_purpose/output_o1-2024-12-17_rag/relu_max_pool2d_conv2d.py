import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_naive_kernel(
    inp_ptr, w_ptr, bias_ptr, out_ptr,
    B, C_in, H_in, W_in,
    K_h, K_w, C_out,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    has_bias: tl.constexpr
):
    # Each program handles one output element (b, co, h_out, w_out).
    # We'll launch a 4D grid: (B, C_out, H_out, W_out).
    b_idx = tl.program_id(0)
    co_idx = tl.program_id(1)
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)

    # Compute valid ranges
    H_out = (H_in + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1
    if (h_out >= H_out) or (w_out >= W_out):
        return

    # Output offset
    out_idx = (b_idx * C_out * H_out * W_out) + (co_idx * H_out * W_out) + (h_out * W_out) + w_out

    # Compute input start
    h_in_center = h_out * stride_h - pad_h
    w_in_center = w_out * stride_w - pad_w

    acc = 0.0
    for ci in range(C_in):
        for kh in range(K_h):
            for kw in range(K_w):
                h_in = h_in_center + kh * dil_h
                w_in = w_in_center + kw * dil_w
                valid = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
                if valid:
                    inp_idx = b_idx * (C_in * H_in * W_in) + ci * (H_in * W_in) + h_in * W_in + w_in
                    w_idx = co_idx * (C_in * K_h * K_w) + ci * (K_h * K_w) + kh * K_w + kw
                    inp_val = tl.load(inp_ptr + inp_idx)
                    w_val = tl.load(w_ptr + w_idx)
                    acc += inp_val * w_val
    if has_bias:
        bias_val = tl.load(bias_ptr + co_idx)
        acc += bias_val

    tl.store(out_ptr + out_idx, acc)


@triton.jit
def maxpool2d_naive_kernel(
    inp_ptr, out_ptr,
    B, C_out, H_in, W_in,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w
):
    # Each program handles one element in (b, c, h_out, w_out) space.
    # We'll launch a 4D grid: (B, C_out, H_out, W_out).
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)

    # Compute valid range for the output shape
    H_out = (H_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
    if (h_out >= H_out) or (w_out >= W_out):
        return

    h_in_start = h_out * stride_h - pad_h
    w_in_start = w_out * stride_w - pad_w

    out_idx = (b_idx * C_out * H_out * W_out) + (c_idx * H_out * W_out) + (h_out * W_out) + w_out

    max_val = float('-inf')
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            h_in = h_in_start + kh * dil_h
            w_in = w_in_start + kw * dil_w
            valid = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
            if valid:
                inp_idx = (b_idx * C_out * H_in * W_in) + (c_idx * H_in * W_in) + h_in * W_in + w_in
                val = tl.load(inp_ptr + inp_idx)
                if val > max_val:
                    max_val = val

    tl.store(out_ptr + out_idx, max_val)


@triton.jit
def relu_naive_kernel(
    inp_ptr, out_ptr,
    N
):
    # 1D grid, each program_id handles a range of elements
    idx = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = idx < N
    val = tl.load(inp_ptr + idx, mask=mask)
    val = tl.where(val > 0, val, 0.0)
    tl.store(out_ptr + idx, val, mask=mask)


def relu_max_pool2d_conv2d(
    input, weight, bias=None,
    conv_stride=1,
    conv_padding=0,
    conv_dilation=1,
    conv_groups=1,
    pool_kernel_size=2,
    pool_stride=None,
    pool_padding=0,
    pool_dilation=1,
    pool_ceil_mode=False,
    inplace=False
):
    """
    Applies a 2D convolution over the input tensor, followed by max pooling and then applies ReLU element-wise.
    input: (minibatch, in_channels, iH, iW)
    weight: (out_channels, in_channels/groups, kH, kW)
    bias: Optional, (out_channels)
    """
    assert input.device.type == "cuda", "Input tensor must be on CUDA device"
    assert weight.device.type == "cuda", "Weight tensor must be on CUDA device"
    if bias is not None:
        assert bias.device.type == "cuda", "Bias tensor must be on CUDA device"
    device = input.device

    # Parse strides/padding/dilation
    if isinstance(conv_stride, int):
        stride_h, stride_w = conv_stride, conv_stride
    else:
        stride_h, stride_w = conv_stride

    if isinstance(conv_padding, int):
        pad_h, pad_w = conv_padding, conv_padding
    else:
        pad_h, pad_w = conv_padding

    if isinstance(conv_dilation, int):
        dil_h, dil_w = conv_dilation, conv_dilation
    else:
        dil_h, dil_w = conv_dilation

    # For simplicity, assume conv_groups == 1 in this example
    B, C_in, H_in, W_in = input.shape
    out_channels, _, K_h, K_w = weight.shape

    # Convolution output shape
    H_out_conv = (H_in + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out_conv = (W_in + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1

    # Allocate output for conv2d
    conv_out = torch.empty((B, out_channels, H_out_conv, W_out_conv), device=device, dtype=input.dtype)

    # Launch conv2d kernel
    has_bias = 1 if (bias is not None) else 0
    grid_conv = (B, out_channels, H_out_conv, W_out_conv)
    conv2d_naive_kernel[grid_conv](
        inp_ptr=input, w_ptr=weight, bias_ptr=bias if bias is not None else weight,  # dummy if no bias
        out_ptr=conv_out,
        B=B, C_in=C_in, H_in=H_in, W_in=W_in,
        K_h=K_h, K_w=K_w, C_out=out_channels,
        stride_h=stride_h, stride_w=stride_w,
        pad_h=pad_h, pad_w=pad_w,
        dil_h=dil_h, dil_w=dil_w,
        has_bias=has_bias
    )

    # Now perform max pool2d
    if pool_stride is None:
        pool_stride = pool_kernel_size
    if isinstance(pool_kernel_size, int):
        pool_kh, pool_kw = pool_kernel_size, pool_kernel_size
    else:
        pool_kh, pool_kw = pool_kernel_size
    if isinstance(pool_stride, int):
        pool_stride_h, pool_stride_w = pool_stride, pool_stride
    else:
        pool_stride_h, pool_stride_w = pool_stride
    if isinstance(pool_padding, int):
        pool_pad_h, pool_pad_w = pool_padding, pool_padding
    else:
        pool_pad_h, pool_pad_w = pool_padding
    if isinstance(pool_dilation, int):
        pool_dil_h, pool_dil_w = pool_dilation, pool_dilation
    else:
        pool_dil_h, pool_dil_w = pool_dilation

    conv_out_shape = conv_out.shape
    _, C_out, H_conv, W_conv = conv_out_shape
    if pool_ceil_mode:
        H_out_pool = (H_conv + 2 * pool_pad_h - pool_dil_h * (pool_kh - 1) - 1 + pool_stride_h - 1) // pool_stride_h + 1
        W_out_pool = (W_conv + 2 * pool_pad_w - pool_dil_w * (pool_kw - 1) - 1 + pool_stride_w - 1) // pool_stride_w + 1
    else:
        H_out_pool = (H_conv + 2 * pool_pad_h - pool_dil_h * (pool_kh - 1) - 1) // pool_stride_h + 1
        W_out_pool = (W_conv + 2 * pool_pad_w - pool_dil_w * (pool_kw - 1) - 1) // pool_stride_w + 1

    pool_out = torch.empty((B, C_out, H_out_pool, W_out_pool), device=device, dtype=input.dtype)

    grid_pool = (B, C_out, H_out_pool, W_out_pool)
    maxpool2d_naive_kernel[grid_pool](
        inp_ptr=conv_out, out_ptr=pool_out,
        B=B, C_out=C_out, H_in=H_conv, W_in=W_conv,
        kernel_h=pool_kh, kernel_w=pool_kw,
        stride_h=pool_stride_h, stride_w=pool_stride_w,
        pad_h=pool_pad_h, pad_w=pool_pad_w,
        dil_h=pool_dil_h, dil_w=pool_dil_w
    )

    # Finally apply ReLU
    # Optionally in-place
    final_out = pool_out if inplace else pool_out.clone()
    N_relu = final_out.numel()
    grid_relu = ( (N_relu + 255) // 256, )
    relu_naive_kernel[grid_relu](
        inp_ptr=pool_out if inplace else pool_out,
        out_ptr=final_out,
        N=N_relu
    )

    return final_out
