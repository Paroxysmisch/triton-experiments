import torch
import triton
import triton.language as tl

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, IC, IH, IW, OC, OH, OW,
    KH, KW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    b_oc = tl.program_id(0)
    oh_ow = tl.program_id(1)

    # Decompose b_oc into batch + out_channel
    b_idx = b_oc // OC
    oc_idx = b_oc % OC

    # Decompose oh_ow into output height + output width
    oh_block = BLOCK_M * oh_ow
    oh_offsets = tl.arange(0, BLOCK_M)
    ow_offsets = tl.arange(0, BLOCK_N)
    # We'll compute for a tile [OH_tile x OW_tile], but here we just do a 1D approach on oh, then broadcast to ow
    # This is a simplistic approach to keep code shorter.

    oh = oh_block + oh_offsets
    ow = ow_offsets  # We'll treat the second dimension differently below

    in_ptr_b = input_ptr + b_idx * IC * IH * IW
    w_ptr_oc = weight_ptr + oc_idx * IC * KH * KW
    out_ptr_b = output_ptr + b_idx * OC * OH * OW

    # Each thread handles a small tile in the output
    oh_mask = oh < OH
    tmp_out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For each oh, we iterate over each ow in a loop
    for i_ow in range(0, OW, BLOCK_N):
        ow_val = i_ow + ow
        ow_mask = ow_val < OW
        mask_2d = oh_mask[:, None] & ow_mask[None, :]

        oh_broad = oh[:, None]
        ow_broad = ow_val[None, :]

        out_val = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # Convolution sum
        for ic_idx in range(IC):
            for kh_idx in range(KH):
                for kw_idx in range(KW):
                    in_h = oh_broad * stride_h + kh_idx * dil_h - pad_h
                    in_w = ow_broad * stride_w + kw_idx * dil_w - pad_w
                    valid_h = (in_h >= 0) & (in_h < IH)
                    valid_w = (in_w >= 0) & (in_w < IW)
                    valid_mask = mask_2d & valid_h & valid_w
                    in_ofs = (
                        ic_idx * IH * IW
                        + in_h * IW
                        + in_w
                    )
                    w_ofs = (
                        ic_idx * KH * KW
                        + kh_idx * KW
                        + kw_idx
                    )
                    val_i = tl.load(in_ptr_b + in_ofs, mask=valid_mask, other=0.0)
                    val_w = tl.load(w_ptr_oc + w_ofs)
                    out_val += val_i * val_w

        if bias_ptr != 0:
            bias_val = tl.load(bias_ptr + oc_idx)
            out_val += bias_val

        # Write partial tile
        out_ofs = oc_idx * OH * OW + oh_broad * OW + ow_broad
        tl.store(out_ptr_b + out_ofs, out_val, mask=mask_2d)

@triton.jit
def _pixel_shuffle_kernel(
    input_ptr, output_ptr,
    B, OC, OH, OW,
    upscale_factor,
    BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elems = B * OC * OH * OW
    mask = idx < total_elems

    # Compute current location in (B, OC, OH, OW)
    b = idx // (OC * OH * OW)
    rem = idx % (OC * OH * OW)
    oc = rem // (OH * OW)
    rem = rem % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # Pixel shuffle formula
    r = upscale_factor
    # Original oc = new_oc * r^2 + sub_idx
    new_oc = oc // (r * r)
    sub_idx = oc % (r * r)
    sub_oh = sub_idx // r
    sub_ow = sub_idx % r

    new_oh = oh * r + sub_oh
    new_ow = ow * r + sub_ow

    # Output shape is (B, new_OC=OC/(r*r), new_OH=OH*r, new_OW=OW*r)
    new_OC = OC // (r * r)
    new_OH = OH * r
    new_OW = OW * r

    old_ofs = idx
    new_ofs = (b * new_OC * new_OH * new_OW) + (new_oc * new_OH * new_OW) + (new_oh * new_OW) + new_ow

    val = tl.load(input_ptr + old_ofs, mask=mask, other=0.0)
    tl.store(output_ptr + new_ofs, val, mask=mask)

def pixel_shuffle_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    upscale_factor=2
) -> torch.Tensor:
    # Shapes
    B, IC, IH, IW = input.shape
    OC, IC_g, KH, KW = weight.shape
    assert IC_g * groups == IC, "Mismatch in in_channels and groups"
    assert OC % (upscale_factor * upscale_factor) == 0, "out_channels must be divisible by upscale_factor^2"

    # Compute output dimensions for convolution
    OH = ((IH + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1
    OW = ((IW + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1

    device = input.device
    assert device.type == "cuda", "Must be on CUDA device"
    input = input.contiguous()
    weight = weight.contiguous()
    bias_tensor = torch.zeros(OC, device=device) if bias is None else bias.contiguous()

    out_conv = torch.empty((B, OC, OH, OW), dtype=input.dtype, device=device)

    # Launch convolution kernel (naive tiling)
    BLOCK_M = 1
    BLOCK_N = 1
    grid_convolution = (B * OC, (OH + BLOCK_M - 1) // BLOCK_M)
    _conv2d_kernel[grid_convolution](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias_tensor if bias is not None else 0,
        output_ptr=out_conv,
        B=B,
        IC=IC,
        IH=IH,
        IW=IW,
        OC=OC,
        OH=OH,
        OW=OW,
        KH=KH,
        KW=KW,
        stride_h=stride,
        stride_w=stride,
        pad_h=padding,
        pad_w=padding,
        dil_h=dilation,
        dil_w=dilation,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )

    # Reshape for pixel shuffle
    out_shuffle = torch.empty(
        (B, OC // (upscale_factor * upscale_factor), OH * upscale_factor, OW * upscale_factor),
        dtype=out_conv.dtype, device=device
    )

    BLOCK_SIZE = 256
    total_elems = B * OC * OH * OW
    grid_shuffle = ((total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _pixel_shuffle_kernel[grid_shuffle](
        input_ptr=out_conv,
        output_ptr=out_shuffle,
        B=B,
        OC=OC,
        OH=OH,
        OW=OW,
        upscale_factor=upscale_factor,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out_shuffle
