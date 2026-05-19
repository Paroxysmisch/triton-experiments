import triton
import triton.language as tl
import torch
from typing import Optional, Union, Tuple

@triton.jit
def _gelu_conv2d_kernel(
    INPUT,              # pointer to input data
    WEIGHT,             # pointer to weight data
    BIAS,               # pointer to bias data (or 0 if None)
    OUTPUT,             # pointer to output data
    stride_h, stride_w, # convolution stride
    pad_h, pad_w,       # padding
    dil_h, dil_w,       # dilation
    in_channels,        # total number of input channels
    out_channels,       # total number of output channels
    kernel_h,           # kernel height
    kernel_w,           # kernel width
    in_h, in_w,         # input height/width
    out_h, out_w,       # output height/width
    groups,             # number of groups
    MISC0,              # (N << 32) | approximate_mode (0 = none, 1 = tanh)
    # N = batch_size, approximate_mode is either 0 or 1
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    # Decode approximate mode and batch size
    approximate_mode = MISC0 & 0xffffffff
    N = MISC0 >> 32

    # Program IDs for 3D launch:
    #   pid_z -> merges batch + out_channel,
    #   pid_y -> partial tile in H dimension of output,
    #   pid_x -> partial tile in W dimension of output
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)

    # Decompose pid_z into batch index (b) and out_channel (oc) index
    b = pid_z // out_channels
    oc = pid_z % out_channels

    # Compute tile indices
    out_y_start = pid_y * BLOCK_H
    out_x_start = pid_x * BLOCK_W

    # Create ranges for the block in H/W dimension
    r_y = out_y_start + tl.arange(0, BLOCK_H)
    r_x = out_x_start + tl.arange(0, BLOCK_W)

    # We'll clamp r_y, r_x to the valid region
    # so that we don't do out-of-bounds writes
    mask_y = r_y < out_h
    mask_x = r_x < out_w

    # Create 2D meshgrid for indexing
    # shape: (BLOCK_H, BLOCK_W)
    Y, X = tl.meshgrid(r_y, r_x)

    # Initialize accumulator with 0
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Each group has out_channels // groups filters and in_channels // groups channels
    # Identify which group "oc" belongs to
    group_id = oc // (out_channels // groups)
    # in_channels within this group
    in_c_start = group_id * (in_channels // groups)
    in_c_end   = in_c_start + (in_channels // groups)

    # For each channel c in the group
    # we sum up the convolution
    for c in range(in_c_start, in_c_end):
        # pointer to the filter for (oc, c)
        # filter offset for the oc-th filter, c-th input channel
        filter_oc_offset = oc * (in_channels // groups) * kernel_h * kernel_w
        filter_c_offset = (c - in_c_start) * kernel_h * kernel_w
        w_offset = filter_oc_offset + filter_c_offset

        # For each (ky, kx) in kernel
        for ky in range(kernel_h):
            for kx in range(kernel_w):
                # compute input spatial location
                in_y = stride_h * Y + dil_h * ky - pad_h
                in_x = stride_w * X + dil_w * kx - pad_w

                # build mask for valid in-bounds (in_y, in_x)
                valid_y = (in_y >= 0) & (in_y < in_h)
                valid_x = (in_x >= 0) & (in_x < in_w)
                valid = valid_y & valid_x & mask_y & mask_x

                # gather input
                in_offset = (b * in_channels + c) * (in_h * in_w)
                gather_idx = in_offset + in_y * in_w + in_x
                inp = tl.load(INPUT + gather_idx, mask=valid, other=0.0)

                # gather weight
                w_idx = w_offset + ky * kernel_w + kx
                w_val = tl.load(WEIGHT + w_idx)

                # accumulate
                acc += inp * w_val

    # Optionally add bias
    if BIAS != 0:
        bias_val = tl.load(BIAS + oc)
        acc += bias_val

    # Now apply GELU
    # approximate_mode: 0 => none, 1 => tanh-based approximation
    if approximate_mode == 0:
        # exact: GELU(x) = x * phi(x)
        # phi(x) ~ 0.5 * [1 + erf(x / sqrt(2))]
        # We'll use an approximation for erf for demonstration or rely on math builtin
        # For better numerical stability, one might implement a more robust version.
        # We'll do a direct approximation of erf using tl.libdevice.erff().
        sqrt_2 = 1.4142135623730951
        acc_fp32 = acc.to(tl.float32)
        # phi(x) = 0.5*(1 + erf(x/sqrt(2)))
        cdf = 0.5 * (1.0 + tl.libdevice.erff(acc_fp32 / sqrt_2))
        acc = acc_fp32 * cdf
    else:
        # approximate_mode == 1 => tanh-based
        # GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ))
        sqrt_2_over_pi = 0.7978845608
        alpha = 0.044715
        x = acc.to(tl.float32)
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + alpha * x_cubed)
        th = tl.libdevice.tanh(inner)
        acc = 0.5 * x * (1.0 + th)

    # Store result
    out_offset = (b * out_channels + oc) * (out_h * out_w)
    store_idx = out_offset + Y * out_w + X
    tl.store(OUTPUT + store_idx, acc, mask=mask_y[:, None] & mask_x[None, :])


def gelu_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    approximate: str = 'none',
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Applies a 2D convolution over an input tensor, then applies GELU activation.
    This function is backed by a Triton kernel.
    """
    # Ensure stride, padding, dilation are tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Handle 'valid' or 'same' padding:
    # For demonstration we'll interpret:
    #   'valid' => no padding
    #   'same' => add padding so output shape is same as input shape (when stride=1)
    # If user wants a more robust approach, they'd have to expand this logic.
    if isinstance(padding, str):
        if padding.lower() == 'valid':
            padding = (0, 0)
        elif padding.lower() == 'same':
            # only well-defined if stride=1
            # for demonstration we compute same padding for kernel
            kH, kW = weight.shape[2], weight.shape[3]
            pad_h = (kH - 1) // 2
            pad_w = (kW - 1) // 2
            padding = (pad_h, pad_w)
        else:
            raise ValueError("Unsupported padding string: {}".format(padding))

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # Extract shapes
    B, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Compute output height/width
    # out_height = floor((in_h + 2*pad_h - dil_h*(kernel_h-1) - 1)/stride_h + 1)
    # out_width  = floor((in_w + 2*pad_w - dil_w*(kernel_w-1) - 1)/stride_w + 1)
    out_h = (in_h + 2*pad_h - dil_h*(kernel_h - 1) - 1)//stride_h + 1
    out_w = (in_w + 2*pad_w - dil_w*(kernel_w - 1
