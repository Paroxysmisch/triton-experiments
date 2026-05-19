import torch
import triton
import triton.language as tl

@triton.jit
def _conv2d_add_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr,
    out_ptr,
    N, C, H, W,
    OC, G,
    KH, KW,
    strideH, strideW,
    padH, padW,
    dilH, dilW,
    alpha,
    has_bias: tl.constexpr,
    has_other: tl.constexpr,
    OTHER_IS_SCALAR: tl.constexpr,
    BLOCK_M: tl.constexpr,  # number of output elements in H dimension per block
    BLOCK_N: tl.constexpr   # number of output elements in W dimension per block
):
    # batch index, out_channels index
    n = tl.program_id(0)
    oc = tl.program_id(1)

    # block indices for height/width
    h_block = tl.program_id(2)
    w_block = tl.program_id(3)

    # offsets within the block
    hm = tl.arange(0, BLOCK_M)
    wn = tl.arange(0, BLOCK_N)

    out_h_start = h_block * BLOCK_M
    out_w_start = w_block * BLOCK_N

    # final index position for H/W
    oh = out_h_start + hm
    ow = out_w_start + wn

    # mask to check if indices are valid in the output
    h_mask = oh < ((H + 2*padH - dilH*(KH-1) - 1)//strideH + 1)
    w_mask = ow < ((W + 2*padW - dilW*(KW-1) - 1)//strideW + 1)
    full_mask = h_mask[:, None] & w_mask[None, :]

    # compute pointer offsets
    # out_ptr shape: [N, OC, outH, outW]
    outH = (H + 2*padH - dilH*(KH-1) - 1)//strideH + 1
    outW = (W + 2*padW - dilW*(KW-1) - 1)//strideW + 1

    # each thread block calculates partial sums of the convolution
    # group offset for out_channels
    g_id = oc // (OC // G)
    c_start = g_id * (C // G)
    c_end = c_start + (C // G)

    # create accumulator for partial sums
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over input channels in this group
    for c_ in range(c_start, c_end):
        # loop over kernel height and width
        for kh in range(KH):
            for kw in range(KW):
                # compute input coords
                in_h = oh*strideH - padH + kh*dilH
                in_w = ow*strideW - padW + kw*dilW

                valid_h = (in_h >= 0) & (in_h < H)
                valid_w = (in_w >= 0) & (in_w < W)
                valid_in = valid_h[:, None] & valid_w[None, :]

                # load input if valid
                inp_offset = (
                    n*C*H*W
                    + c_*H*W
                )
                # positions for each element
                in_pos = inp_offset + in_h[:, None]*W + in_w[None, :]

                input_vals = tl.where(valid_in & full_mask, tl.load(input_ptr + in_pos, mask=valid_in & full_mask, other=0.0), 0.0)

                # get weight
                wt_offset = (
                    oc*(C//G)*KH*KW
                    + (c_ - c_start)*KH*KW
                )
                wgt_pos = wt_offset + kh*KW + kw
                weight_val = tl.load(weight_ptr + wgt_pos)

                acc += input_vals * weight_val

    # add bias if present
    if has_bias:
        bias_val = tl.load(bias_ptr + oc)
        acc += bias_val

    # add alpha * other if present
    if has_other:
        if OTHER_IS_SCALAR:
            acc += alpha * tl.load(other_ptr)
        else:
            # shape [N, OC, outH, outW]
            otherH = outH
            otherW = outW
            # position into 'other'
            off_out = n*OC*otherH*otherW + oc*otherH*otherW + oh[:, None]*otherW + ow[None, :]
            other_vals = tl.where(
                full_mask,
                tl.load(other_ptr + off_out, mask=full_mask, other=0.0),
                0.0
            )
            acc += alpha * other_vals

    # store result
    out_off = n*OC*outH*outW + oc*outH*outW + oh[:, None]*outW + ow[None, :]
    tl.store(out_ptr + out_off, acc, mask=full_mask)


def conv2d_add(input, weight, bias=None, other=None,
               stride=1, padding=0, dilation=1,
               groups=1, alpha=1, out=None):
    # Validate shapes
    N, C, H, W = input.shape
    OC, IC_per_group, KH, KW = weight.shape
    assert (C % groups == 0) and (OC % groups == 0), "Groups must divide input and output channels."
    assert IC_per_group * groups == C, "Weight shape mismatch with input channels."
    # handle stride, padding, dilation as tuple
    if isinstance(stride, int):
        strideH, strideW = stride, stride
    else:
        strideH, strideW = stride

    if isinstance(padding, int):
        padH, padW = padding, padding
    elif padding == 'valid':
        padH, padW = 0, 0
    elif padding == 'same':
        # 'same' padding for stride=1, approximate
        padH = (KH - 1) // 2
        padW = (KW - 1) // 2
    else:
        padH, padW = padding

    if isinstance(dilation, int):
        dilH, dilW = dilation, dilation
    else:
        dilH, dilW = dilation

    # compute output shape
    outH = (H + 2*padH - dilH*(KH-1) - 1)//strideH + 1
    outW = (W + 2*padW - dilW*(KW-1) - 1)//strideW + 1
    if out is None:
        out = torch.empty((N, OC, outH, outW), device=input.device, dtype=input.dtype)

    # set up kernel arguments
    has_bias = (bias is not None)
    has_other = (other is not None)

    OTHER_IS_SCALAR = False
    other_t = None
    if has_other:
        if isinstance(other, (int, float)):
            # allocate a scalar tensor on device
            other_t = torch.tensor(other, device=input.device, dtype=input.dtype)
            OTHER_IS_SCALAR = True
        elif isinstance(other, torch.Tensor):
            other_t = other
        else:
            raise RuntimeError("Unsupported type for 'other'")

    # define block size for output tile
    BLOCK_M = 8
    BLOCK_N = 8

    # grid: each block processes (BLOCK_M, BLOCK_N) within output for each (N, oc)
    grid = (
        N,
        OC,
        (outH + BLOCK_M - 1) // BLOCK_M,
        (outW + BLOCK_N - 1) // BLOCK_N,
    )

    _conv2d_add_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias if has_bias else input,  # dummy if no bias
        other_ptr=other_t if has_other else input,  # dummy if no other
        out_ptr=out,
        N=N, C=C, H=H, W=W,
        OC=OC, G=groups,
        KH=KH, KW=KW,
        strideH=strideH, strideW=strideW,
        padH=padH, padW=padW,
        dilH=dilH, dilW=dilW,
        alpha=alpha,
        has_bias=has_bias,
        has_other=has_other,
        OTHER_IS_SCALAR=OTHER_IS_SCALAR,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )

    return out
