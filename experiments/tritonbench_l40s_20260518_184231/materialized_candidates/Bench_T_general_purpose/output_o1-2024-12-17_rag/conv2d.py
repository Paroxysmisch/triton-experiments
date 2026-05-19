import torch
import triton
import triton.language as tl

@triton.jit
def _conv2d_kernel(
    input_ptr,                 # (B, IC, IH, IW)
    weight_ptr,                # (OC, IC/groups, KH, KW)
    bias_ptr,                  # (OC) or None
    output_ptr,                # (B, OC, OH, OW)
    B, IC, IH, IW,
    OC, KH, KW,
    OH, OW,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    """
    A naive Triton 2D convolution kernel. For each output position,
    it accumulates the product of input patches and the corresponding
    weight filter, adding bias if provided. Demonstration purposes only.
    """

    # Compute the output index we are responsible for
    b_idx = tl.program_id(0)  # batch index
    oc_idx = tl.program_id(1)  # output channel block
    oh_block = tl.program_id(2)  # output height block

    # Each block iterates over a rectangular region (BLOCK_M x BLOCK_N) in (height, width)
    ow_block = tl.arange(0, BLOCK_N)
    oh_global = oh_block * BLOCK_M + tl.arange(0, BLOCK_M)

    # Expand dims to shape (BLOCK_M, BLOCK_N)
    oh_exp = oh_global[:, None]
    ow_exp = ow_block[None, :]

    # Compute valid mask
    valid_h = (oh_exp < OH)
    valid_w = (ow_exp < OW)
    valid = valid_h & valid_w

    # Offset in output memory for (B, OC, OH, OW)
    oc_offset = oc_idx
    out_offset = (b_idx * OC * OH * OW) + oc_offset * (OH * OW) + oh_exp * OW + ow_exp

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Each oc block has size 1 for demonstration, so oc_idx < OC
    if oc_idx < OC:
        # For the group the current OC belongs to
        oc_group = oc_idx // (OC // groups)
        # Start and end for input channels in this group
        ic_start = oc_group * (IC // groups)
        ic_end = ic_start + (IC // groups)

        # Load bias if provided
        if tl.pointer_is_valid(bias_ptr):
            b_val = tl.load(bias_ptr + oc_idx)
        else:
            b_val = 0.0

        # Compute output's (oh, ow) in input space
        in_h = oh_exp * stride_h - pad_h
        in_w = ow_exp * stride_w - pad_w

        # Accumulate convolution
        for ic in range(ic_start, ic_end):
            for kh in range(KH):
                for kw in range(KW):
                    # Calculate the input indices
                    h_index = in_h + kh * dilation_h
                    w_index = in_w + kw * dilation_w

                    # Check boundaries
                    in_bounds = (h_index >= 0) & (h_index < IH) & (w_index >= 0) & (w_index < IW) & valid
                    # If in bounds, load input
                    input_offset = (
                        b_idx * (IC * IH * IW)
                        + ic * (IH * IW)
                        + h_index * IW
                        + w_index
                    )
                    inp_val = tl.where(
                        in_bounds,
                        tl.load(input_ptr + input_offset, mask=in_bounds, other=0.0),
                        0.0
                    )

                    # Load weight
                    w_offset = (
                        oc_idx * (IC // groups) * (KH * KW)
                        + (ic - ic_start) * (KH * KW)
                        + kh * KW
                        + kw
                    )
                    w_val = tl.load(weight_ptr + w_offset)

                    # Accumulate
                    acc += inp_val * w_val

        # Add bias
        acc = acc + b_val
        # Store results
        tl.store(output_ptr + out_offset, acc, mask=valid)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Applies a 2D convolution over an input image composed of several input planes.
    Supports TensorFloat32. May select a nondeterministic algorithm on CUDA
    with CuDNN for performance. Supports complex data types.

    Args:
        input (torch.Tensor): Input tensor of shape (minibatch, in_channels, iH, iW).
        weight (torch.Tensor): Filters of shape (out_channels, in_channels/groups, kH, kW).
        bias (torch.Tensor, optional): Optional bias tensor of shape (out_channels). Default: None.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int or tuple or str, optional): Padding. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of groups to split into. Default: 1.

    Returns:
        torch.Tensor: Output tensor of shape (minibatch, out_channels, oH, oW).
    """

    # Assertions and shape checks
    device = input.device
    assert device.type == "cuda", "Input tensor must be on a CUDA device for Triton ops."
    assert weight.device == device, "Weight must be on the same device as input."
    if bias is not None:
        assert bias.device == device, "Bias must be on the same device as input."

    B, IC, IH, IW = input.shape
    OC, ICg, KH, KW = weight.shape
    assert ICg * groups == IC, "Channels in weight must match input channels divided by groups."
    # Convert stride, padding, dilation to tuple
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h, pad_w = padding, padding
    elif isinstance(padding, tuple):
        pad_h, pad_w = padding
    elif isinstance(padding, str) and padding.lower() in ['same', 'valid']:
        # Minimal shape-based logic for demonstration
        if padding.lower() == 'same':
            pad_h = (KH - 1) // 2
            pad_w = (KW - 1) // 2
        else:  # 'valid'
            pad_h, pad_w = 0, 0
    else:
        raise ValueError("Unsupported padding format.")
    if isinstance(dilation, int):
        dilation_h, dilation_w = dilation, dilation
    else:
        dilation_h, dilation_w = dilation

    # Determine output spatial size
    OH = ((IH + 2 * pad_h - dilation_h * (KH - 1) - 1) // stride_h) + 1
    OW = ((IW + 2 * pad_w - dilation_w * (KW - 1) - 1) // stride_w) + 1

    # Prepare output
    out_shape = (B, OC, OH, OW)
    output = torch.empty(out_shape, dtype=input.dtype, device=device)

    # Setup Triton grid
    BLOCK_M = 8   # block height
    BLOCK_N = 8   # block width
    grid_b = B
    grid_oc = OC
    grid_oh = (OH + BLOCK_M - 1) // BLOCK_M
    # We'll combine oh/ow loops in the kernel
    # by letting oh be a separate dimension and ow be a range in kernel
    grid = (grid_b, grid_oc, grid_oh)

    bias_ptr = bias if bias is not None else None

    _conv2d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias_ptr,
        output_ptr=output,
        B=B,
        IC=IC,
        IH=IH,
        IW=IW,
        OC=OC,
        KH=KH,
        KW=KW,
        OH=OH,
        OW=OW,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        groups=groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return output
