import torch
import triton
import triton.language as tl

@triton.jit
def _relu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, inC, iH, iW,
    outC, kH, kW,
    strideH, strideW,
    padH, padW,
    dilH, dilW,
    groups,
    OUT_H, OUT_W,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    batch_channel = tl.program_id(0)
    oh_start = tl.program_id(1) * BLOCK_H
    ow_start = tl.program_id(2) * BLOCK_W

    # Decode batch and output channel
    b_idx = batch_channel // outC
    oc_idx = batch_channel % outC

    # Create a range of output height/width indices
    oh_offsets = oh_start + tl.arange(0, BLOCK_H)
    ow_offsets = ow_start + tl.arange(0, BLOCK_W)

    # Create a 2D mesh of (OH, OW) for the block
    OH, OW = tl.meshgrid(oh_offsets, ow_offsets)

    # Check boundaries
    valid_H = OH < OUT_H
    valid_W = OW < OUT_W
    mask = valid_H & valid_W

    # Initialize accumulator
    out_vals = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Grouped convolution calculations
    group_idx = oc_idx // (outC // groups)
    c_per_group = inC // groups
    w_offset = oc_idx * c_per_group * kH * kW + group_idx * c_per_group * kH * kW - group_idx * c_per_group * kH * kW

    # Compute convolution
    for ic in range(c_per_group):
        for kh in range(kH):
            for kw in range(kW):
                # Compute input coordinate with dilation
                inp_h = OH * strideH + (kh * dilH) - padH
                inp_w = OW * strideW + (kw * dilW) - padW

                # Compute pointer offsets
                ic_idx = group_idx * c_per_group + ic
                in_ptr_offset = (
                    b_idx * inC * iH * iW
                    + ic_idx * iH * iW
                    + inp_h * iW
                    + inp_w
                )

                # Load input values, apply mask for validity
                in_mask = (inp_h >= 0) & (inp_h < iH) & (inp_w >= 0) & (inp_w < iW) & mask
                in_vals = tl.load(input_ptr + in_ptr_offset, mask=in_mask, other=0.0)

                # Load weight
                w_ptr_offset = w_offset + ic * kH * kW + kh * kW + kw
                w_val = tl.load(weight_ptr + w_ptr_offset)

                # Accumulate
                out_vals += in_vals * w_val

    # Add bias if not None
    if tl.dynamic_range(bias_ptr) > 0:
        bias_val = tl.load(bias_ptr + oc_idx)
        out_vals = out_vals + bias_val

    # ReLU
    out_vals = tl.maximum(out_vals, 0.0)

    # Store result
    out_offset = (
        b_idx * outC * OUT_H * OUT_W
        + oc_idx * OUT_H * OUT_W
        + OH * OUT_W
        + OW
    )
    tl.store(output_ptr + out_offset, out_vals, mask=mask)


def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    # Ensure CUDA
    assert input.is_cuda, "input must be on CUDA"
    assert weight.is_cuda, "weight must be on CUDA"
    if bias is not None:
        assert bias.is_cuda, "bias must be on CUDA"

    # Parse parameters
    B, inC, iH, iW = input.shape
    outC, _, kH, kW = weight.shape

    if isinstance(stride, int):
        strideH, strideW = stride, stride
    else:
        strideH, strideW = stride

    if isinstance(padding, int):
        padH, padW = padding, padding
    else:
        padH, padW = padding

    if isinstance(dilation, int):
        dilH, dilW = dilation, dilation
    else:
        dilH, dilW = dilation

    # Compute output spatial dimensions
    outH = (iH + 2*padH - dilH*(kH-1) - 1)//strideH + 1
    outW = (iW + 2*padW - dilW*(kW-1) - 1)//strideW + 1

    # Create output
    if inplace:
        # If in-place, we'll just create a separate buffer and copy later
        output = input.new_empty((B, outC, outH, outW))
    else:
        output = input.new_empty((B, outC, outH, outW))

    # Handle bias pointer for Triton
    bias_ptr = bias if bias is not None else torch.tensor([], device=input.device)

    # Grid
    BLOCK_H = 8
    BLOCK_W = 8
    grid = (
        B * outC, 
        (outH + BLOCK_H - 1) // BLOCK_H,
        (outW + BLOCK_W - 1) // BLOCK_W,
    )

    _relu_conv2d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias_ptr,
        output_ptr=output,
        B=B,
        inC=inC,
        iH=iH,
        iW=iW,
        outC=outC,
        kH=kH,
        kW=kW,
        strideH=strideH,
        strideW=strideW,
        padH=padH,
        padW=padW,
        dilH=dilH,
        dilW=dilW,
        groups=groups,
        OUT_H=outH,
        OUT_W=outW,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )

    if inplace:
        input.zero_()
        input[..., :outH, :outW].copy_(output)
        return input
    return output
