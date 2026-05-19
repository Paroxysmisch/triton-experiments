import triton
import triton.language as tl

@triton.jit
def _conv2d_kernel(
    input_ptr,      # [B, C, H, W]
    weight_ptr,     # [M, C/groups, R, S]
    bias_ptr,       # [M] or None
    output_ptr,     # [B, M, outH, outW]
    B, M, C, H, W,
    R, S,
    outH, outW,
    strideH, strideW,
    padH, padW,
    dilationH, dilationW,
    groups,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    # program_id's for indexing block of output
    b_idx = tl.program_id(0)
    oc_blk = tl.program_id(1)
    oh_blk = tl.program_id(2)
    ow_blk = tl.program_id(3)

    oc_range = tl.arange(0, BLOCK_M)
    oh_range = tl.arange(0, BLOCK_H)
    ow_range = tl.arange(0, BLOCK_W)

    oc_out = oc_blk * BLOCK_M + oc_range
    oh_out = oh_blk * BLOCK_H + oh_range
    ow_out = ow_blk * BLOCK_W + ow_range

    # Create a 2D expansion for oh_out, ow_out
    oh_out_2d = oh_out[:, None]  # [BLOCK_H, 1]
    ow_out_2d = ow_out[None, :]  # [1, BLOCK_W]

    # Check bounds
    valid_b = b_idx < B
    valid_oc = oc_out < M
    valid_oh = oh_out_2d < outH
    valid_ow = ow_out_2d < outW

    # Prepare accumulators
    output_val = tl.zeros((BLOCK_M, BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Each threadblock processes a sub-tile in output
    if valid_b:
        for g in range(groups):
            # oc_out is in the range [0..M-1], group offset check
            group_start = g * (C // groups)
            # Identify which oc_out belong to current group
            within_group = (oc_out // ((M // groups))) == g
            # Expand group condition
            valid_group_oc = valid_oc & within_group

            # Accumulate convolution
            for c_in in range(C // groups):
                c_idx = group_start + c_in
                for r_k in range(R):
                    for s_k in range(S):
                        # Compute input spatial location
                        in_h = oh_out_2d * strideH - padH + r_k * dilationH
                        in_w = ow_out_2d * strideW - padW + s_k * dilationW
                        # Check in-range
                        in_bounds = (0 <= in_h) & (in_h < H) & (0 <= in_w) & (in_w < W)
                        # Load input if valid
                        if tl.any(in_bounds):
                            # broadcast in_h, in_w for each cell
                            in_h_brd = tl.where(in_bounds, in_h, 0)
                            in_w_brd = tl.where(in_bounds, in_w, 0)
                            # Flatten input index
                            inp_offset = (
                                b_idx * C * H * W
                                + c_idx * H * W
                                + in_h_brd * W
                                + in_w_brd
                            )
                            val_in = tl.load(input_ptr + inp_offset, mask=in_bounds, other=0.0)
                            # Flatten weight index
                            # oc_out has shape [BLOCK_M], c_in is single, r_k, s_k
                            w_offset = (
                                oc_out * (C // groups) * R * S
                                + c_in * R * S
                                + r_k * S
                                + s_k
                            )
                            mask_oc = valid_group_oc
                            val_w = tl.load(weight_ptr + w_offset, mask=mask_oc, other=0.0)
                            # Add to accumulator
                            # Expand val_in to [BLOCK_H, BLOCK_W], val_w to [BLOCK_M,1,1]
                            val_in_f32 = val_in.to(tl.float32)
                            val_w_f32 = val_w[:, None, None].to(tl.float32)
                            output_val += val_in_f32[None, :, :] * val_w_f32

        # Add bias if given
        if bias_ptr != 0:
            b_val = tl.load(bias_ptr + oc_out, mask=valid_oc, other=0.0)
            output_val += b_val[:, None, None]

        # Write out
        # Flatten output index
        out_offset_base = b_idx * M * outH * outW
        for i_oc in range(BLOCK_M):
            if valid_oc[i_oc]:
                for i_oh in range(BLOCK_H):
                    if valid_oh[i_oh, 0]:
                        for i_ow in range(BLOCK_W):
                            if valid_ow[0, i_ow]:
                                out_offset = out_offset_base + (oc_out[i_oc] * outH * outW) \
                                             + (oh_out[i_oh] * outW) \
                                             + ow_out[i_ow]
                                tl.store(output_ptr + out_offset, output_val[i_oc, i_oh, i_ow])

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
    input  : tensor of shape [B, C, H, W]
    weight : tensor of shape [M, C/groups, R, S]
    bias   : optional, shape [M]
    stride : single int or tuple (strideH, strideW)
    padding: single int, tuple (padH, padW), or 'same'/'valid'
    dilation: single int or tuple (dH, dW)
    groups : int
    """
    # Handle tuple or int arguments
    if isinstance(stride, int):
        strideH, strideW = stride, stride
    else:
        strideH, strideW = stride

    if isinstance(padding, int):
        padH, padW = padding, padding
    elif isinstance(padding, str):
        # For simplicity, support only 'valid' or 'same' (no special shape checks)
        if padding.lower() == 'valid':
            padH, padW = 0, 0
        else:  # 'same' 
            # Simple for demonstration: compute pad for "same" as done in many frameworks
            H_in, W_in = input.shape[2], input.shape[3]
            R, S = weight.shape[2], weight.shape[3]
            padH = ((H_in - 1) * strideH + R - H_in) // 2
            padW = ((W_in - 1) * strideW + S - W_in) // 2
    else:
        padH, padW = padding

    if isinstance(dilation, int):
        dilationH, dilationW = dilation, dilation
    else:
        dilationH, dilationW = dilation

    B, C, H, W = input.shape
    M, Cg, R, S = weight.shape
    # Cg should be = C/groups
    outH = (H + 2 * padH - dilationH * (R - 1) - 1) // strideH + 1
    outW = (W + 2 * padW - dilationW * (S - 1) - 1) // strideW + 1

    import torch
    # Convert input, weight, bias to device pointers
    in_ptr = input.contiguous()
    wt_ptr = weight.contiguous()
    if bias is not None:
        bias_ptr = bias.contiguous()
    else:
        # Pass dummy pointer if no bias
        bias_ptr = torch.tensor([], device=in_ptr.device, dtype=in_ptr.dtype)

    # Allocate output
    out = torch.empty((B, M, outH, outW), device=in_ptr.device, dtype=in_ptr.dtype)

    # Launch kernel
    BLOCK_M = 8
    BLOCK_H = 4
    BLOCK_W = 4

    grid = (
        B,                  # batch dimension
        (M + BLOCK_M - 1) // BLOCK_M,   # out_channel blocks
        (outH + BLOCK_H - 1) // BLOCK_H,
        (outW + BLOCK_W - 1) // BLOCK_W
    )

    _conv2d_kernel[grid](
        in_ptr, wt_ptr, bias_ptr if bias is not None else 0, out,
        B, M, C, H, W,
        R, S,
        outH, outW,
        strideH, strideW,
        padH, padW,
        dilationH, dilationW,
        groups,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )

    return out
