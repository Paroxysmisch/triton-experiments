import triton
import triton.language as tl
import math


@triton.jit
def _conv2d_bn_relu_kernel(
    input_ptr, weight_ptr, bias_ptr,
    running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    output_ptr,
    BATCH, IN_CH, IN_H, IN_W,
    OUT_CH, KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PADDING_H, PADDING_W,
    DILATION_H, DILATION_W,
    GROUPS,
    EPS,
    BLOCK_SIZE: tl.constexpr
):
    # Flat index for each output element across (BATCH, OUT_CH, OUT_H, OUT_W)
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Total number of output elements
    out_elems = BATCH * OUT_CH * ((IN_H + 2 * PADDING_H - DILATION_H * (KERNEL_H - 1) - 1) // STRIDE_H + 1) * (
            (IN_W + 2 * PADDING_W - DILATION_W * (KERNEL_W - 1) - 1) // STRIDE_W + 1
    )

    # Mask: valid threads only if offset < out_elems
    mask = offsets < out_elems

    # Compute n, oc, oh, ow from the flat offset
    # out_h = (IN_H + 2*pad_h - dilation_h*(k_h-1) - 1)//stride_h + 1
    out_h = (IN_H + 2 * PADDING_H - DILATION_H * (KERNEL_H - 1) - 1) // STRIDE_H + 1
    out_w = (IN_W + 2 * PADDING_W - DILATION_W * (KERNEL_W - 1) - 1) // STRIDE_W + 1

    n = offsets // (OUT_CH * out_h * out_w)
    rem = offsets % (OUT_CH * out_h * out_w)
    oc = rem // (out_h * out_w)
    rem = rem % (out_h * out_w)
    oh = rem // out_w
    ow = rem % out_w

    # Initialize accumulator for convolution
    conv_val = tl.zeros_like(offsets, dtype=tl.float32)

    # Each output channel belongs to a group of in_channels if groups > 1
    group_idx = oc // (OUT_CH // GROUPS)
    in_ch_start = group_idx * (IN_CH // GROUPS)
    in_ch_end = (group_idx + 1) * (IN_CH // GROUPS)

    # Compute convolution
    for ic in range(in_ch_start, in_ch_end):
        for kh in range(KERNEL_H):
            for kw in range(KERNEL_W):
                # Compute input spatial location
                ih = oh * STRIDE_H + kh * DILATION_H - PADDING_H
                iw = ow * STRIDE_W + kw * DILATION_W - PADDING_W

                # Check boundaries
                in_bounds = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & mask
                if in_bounds:
                    input_idx = (n * IN_CH * IN_H * IN_W) + (ic * IN_H * IN_W) + (ih * IN_W) + iw
                    w_ch = oc % OUT_CH
                    weight_idx = (w_ch * (IN_CH // GROUPS) * KERNEL_H * KERNEL_W) + (
                        (ic - in_ch_start) * KERNEL_H * KERNEL_W
                    ) + (kh * KERNEL_W) + kw
                    val_input = tl.load(input_ptr + input_idx, mask=in_bounds)
                    val_weight = tl.load(weight_ptr + weight_idx, mask=in_bounds)
                    conv_val += val_input * val_weight

    # Add bias if available
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + oc, mask=mask, other=0.0)
        conv_val += bias_val

    # BatchNorm if running_mean, running_var, bn_weight, bn_bias are provided
    if (running_mean_ptr is not None) and (running_var_ptr is not None) and \
       (bn_weight_ptr is not None) and (bn_bias_ptr is not None):
        mean_val = tl.load(running_mean_ptr + oc, mask=mask, other=0.0)
        var_val = tl.load(running_var_ptr + oc, mask=mask, other=0.0)
        w_val = tl.load(bn_weight_ptr + oc, mask=mask, other=1.0)
        b_val = tl.load(bn_bias_ptr + oc, mask=mask, other=0.0)
        inv_std_val = 1.0 / tl.sqrt(var_val + EPS)
        conv_val = (conv_val - mean_val) * inv_std_val * w_val + b_val

    # ReLU
    relu_val = tl.where(conv_val > 0.0, conv_val, 0.0)

    # Store output
    out_idx = offsets
    tl.store(output_ptr + out_idx, relu_val, mask=mask)


def relu_batch_norm_conv2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
    running_mean=None, running_var=None, bn_weight=None, bn_bias=None,
    training=False, momentum=0.1, eps=1e-5, inplace=False
):
    """
    Applies a 2D convolution over the input tensor, followed by batch normalization
    and then applies the ReLU activation function element-wise.
    """
    # Parse args
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Ensure tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # Compute output height/width
    out_h = (in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1

    # Allocate output
    import torch
    output = torch.empty((batch_size, out_channels, out_h, out_w), dtype=torch.float32, device=input.device)

    # Flatten output for kernel
    out_elems = batch_size * out_channels * out_h * out_w

    # Launch Triton kernel
    BLOCK_SIZE = 256
    grid = lambda META: ((out_elems + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'],)
    conv2d_bn_relu_kernel = _conv2d_bn_relu_kernel[grid](
        input, weight, bias if bias is not None else tl.zeros(1, dtype=tl.float32),
        running_mean if running_mean is not None else tl.zeros(out_channels, dtype=tl.float32),
        running_var if running_var is not None else tl.zeros(out_channels, dtype=tl.float32),
        bn_weight if bn_weight is not None else tl.zeros(out_channels, dtype=tl.float32),
        bn_bias if bn_bias is not None else tl.zeros(out_channels, dtype=tl.float32),
        output,
        batch_size, in_channels, in_h, in_w,
        out_channels, kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        groups,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Naive update for running stats in training mode (not implemented in this sample)
    # if training:
    #     <update running_mean, running_var here, e.g. via Welford or EMA>

    return output
