import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B, IC, IH, IW, OC, OH, OW,
    KH, KW,
    strideH, strideW,
    padH, padW,
    dilationH, dilationW,
    groups,
    approximate_tanh: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Naive 2D convolution + GELU in Triton.
    Each program processes a small tile (BLOCK_H x BLOCK_W) of the output
    for a single (batch, out_channel).
    """
    # Current program's batch index and out_channel index
    b = tl.program_id(0)
    oc = tl.program_id(1)

    # Compute the starting indices in the output for this tile
    oh_start = tl.program_id(2) * BLOCK_H
    ow_start = tl.program_id(3) * BLOCK_W

    # Offsets in the tile
    oh_offsets = oh_start + tl.arange(0, BLOCK_H)
    ow_offsets = ow_start + tl.arange(0, BLOCK_W)

    # Create 2D mesh for indexing
    oh_mesh = oh_offsets[:, None]
    ow_mesh = ow_offsets[None, :]

    # Prepare mask for valid output
    out_mask_h = (oh_mesh >= 0) & (oh_mesh < OH)
    out_mask_w = (ow_mesh >= 0) & (ow_mesh < OW)
    out_mask = out_mask_h & out_mask_w

    # Compute ptr offset for out
    out_index = (b * OC * OH * OW) + (oc * OH * OW) + (oh_mesh * OW + ow_mesh)
    # Accumulator for convolution
    conv_sum = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # in_channels per group and out_channels per group
    ic_per_group = IC // groups
    group_idx = oc // (OC // groups)
    # For each input channel in the group
    for ic in range(ic_per_group):
        real_ic = group_idx * ic_per_group + ic
        # For each kernel element
        for kh in range(KH):
            for kw in range(KW):
                # Compute input's spatial location
                ih = oh_mesh * strideH + kh * dilationH - padH
                iw = ow_mesh * strideW + kw * dilationW - padW
                # Check boundary
                valid_h = (ih >= 0) & (ih < IH)
                valid_w = (iw >= 0) & (iw < IW)
                valid = valid_h & valid_w & out_mask

                # Calculate index in input and weight
                in_index = (b * IC * IH * IW) + (real_ic * IH * IW) + (ih * IW + iw)
                w_index = (
                    oc * ic_per_group * KH * KW
                    + ic * KH * KW
                    + kh * KW
                    + kw
                )

                # Load input and weight
                inp_val = tl.where(valid, tl.load(input_ptr + in_index, mask=valid, other=0.0), 0.0)
                w_val = tl.load(weight_ptr + w_index)

                conv_sum += inp_val * w_val

    # Optional bias
    if tl.static_assert(bias_ptr != 0, ""):
        bias_val = tl.load(bias_ptr + oc)
        conv_sum += bias_val

    # Apply GELU
    # approximate_tanh indicates if we use the tanh approximation
    if approximate_tanh:
        # tanh approximation
        x = conv_sum
        x3 = x * x * x
        inner = 0.79788456 * (x + 0.044715 * x3)
        gelu_val = 0.5 * x * (1.0 + tl.math.tanh(inner))
        conv_sum = gelu_val
    else:
        # exact (erf-based) gelu
        # gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt2 = 1.41421356237
        x = conv_sum
        r = 0.5 * x * (1.0 + tl.libdevice.erf(x / sqrt2))
        conv_sum = r

    # Store output
    tl.store(out_ptr + out_index, conv_sum, mask=out_mask)


def gelu_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Optional[torch.Tensor] = None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    approximate: str = 'none',
    out: torch.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies a 2D convolution over the input tensor followed by a GELU activation.
    """
    device = input.device
    assert device.type == 'cuda', "Input tensor must be on a CUDA device."
    assert weight.device == device, "Weight must be on the same device as input."

    B, IC, IH, IW = input.shape
    OC, IC_per_group, KH, KW = weight.shape
    assert IC == IC_per_group * groups, "Mismatch in input channels vs. weight shape/groups."

    # Convert stride, padding, dilation to 2D tuples if needed
    def _to_2d(x):
        return (x, x) if isinstance(x, int) else x

    stride_h, stride_w = _to_2d(stride)
    pad_h, pad_w = 0, 0
    dilation_h, dilation_w = _to_2d(dilation)

    # Handle padding specification
    if isinstance(padding, str):
        # 'valid' or 'same'
        # For simplicity, only handle 'valid' (no padding) or 'same' (pad so output same size)
        if padding == 'valid':
            pad_h, pad_w = 0, 0
        elif padding == 'same':
            # For 'same': (out_size = in_size) => pad = ((in_size - 1)*stride + kernel - in_size) / 2
            pad_h = ((IH - 1) * stride_h + KH - IH) // 2
            pad_w = ((IW - 1) * stride_w + KW - IW) // 2
    elif isinstance(padding, int):
        pad_h, pad_w = padding, padding
    else:
        pad_h, pad_w = padding

    # Compute output height/width
    OH = ((IH + 2 * pad_h - dilation_h * (KH - 1) - 1) // stride_h) + 1
    OW = ((IW + 2 * pad_w - dilation_w * (KW - 1) - 1) // stride_w) + 1

    # Prepare output
    if out is None:
        out = torch.empty((B, OC, OH, OW), dtype=input.dtype, device=device)

    # Move bias to device if needed
    bias_ptr = 0
    if bias is not None:
        bias = bias.to(device)
        bias_ptr = bias.data_ptr()

    # Kernel launch config
    BLOCK_H = 8
    BLOCK_W = 8
    grid = (
        B,                # batch
        OC,               # out channels
        (OH + BLOCK_H - 1) // BLOCK_H,  # blocks in H dimension
        (OW + BLOCK_W - 1) // BLOCK_W,  # blocks in W dimension
    )
    approximate_tanh = (approximate == 'tanh')

    _gelu_conv2d_kernel[grid](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias_ptr,
        out_ptr=out.data_ptr(),
        B=B,
        IC=IC,
        IH=IH,
        IW=IW,
        OC=OC,
        OH=OH,
        OW=OW,
        KH=KH,
        KW=KW,
        strideH=stride_h,
        strideW=stride_w,
        padH=pad_h,
        padW=pad_w,
        dilationH=dilation_h,
        dilationW=dilation_w,
        groups=groups,
        approximate_tanh=approximate_tanh,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    return out
