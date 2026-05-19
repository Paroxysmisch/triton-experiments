import torch
import triton
import triton.language as tl

@triton.jit
def _sigmoid_conv2d_kernel(
    input_ptr,         # float32[B, IN_C, IN_H, IN_W]
    weight_ptr,        # float32[OUT_C, IN_C/groups, K_H, K_W]
    bias_ptr,          # float32[OUT_C] or None
    output_ptr,        # float32[B, OUT_C, OUT_H, OUT_W]
    B, IN_C, IN_H, IN_W,
    OUT_C, K_H, K_W,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    groups,
    out_H, out_W,
    # Block sizes for parallelization
    BLOCK_M: tl.constexpr,  # number of output elements computed along OH/OW dimension
    BLOCK_N: tl.constexpr   # number of output channels or batch slices
):
    """
    Naive 2D convolution + Sigmoid kernel in Triton.
    Each program calculates one tile [BLOCK_M] x [BLOCK_M] over (OH, OW)
    for a slice of batch*channel dimension size [BLOCK_N].
    """
    bid = tl.program_id(0)
    # bid covers the combination of B and OUT_C, we decode which B, which out_channel set we handle
    b = bid // OUT_C
    oc_block = bid % OUT_C

    # program_id(1) will index in out_H/out_W dimension
    oh_block = tl.program_id(1) * BLOCK_M
    ow_block = tl.program_id(2) * BLOCK_M

    # create ranges for local tile
    r_oh = oh_block + tl.arange(0, BLOCK_M)
    r_ow = ow_block + tl.arange(0, BLOCK_M)

    # Define accumulators for each (oh,ow) in the local tile
    # We store intermediates in a 2D array of shape (BLOCK_M, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)

    # Figure out group indexing
    in_c_per_group = IN_C // groups
    group_idx = oc_block // (OUT_C // groups)
    in_c_start = group_idx * in_c_per_group
    in_c_end = in_c_start + in_c_per_group

    # For each position in the kernel, accumulate convolution
    for ic in range(in_c_start, in_c_end):
        for kh in range(K_H):
            for kw in range(K_W):
                # weight index
                w_offset = (
                    oc_block * in_c_per_group * K_H * K_W
                    + (ic - in_c_start) * K_H * K_W
                    + kh * K_W
                    + kw
                )
                w_val = tl.load(weight_ptr + w_offset)

                # compute input coordinates
                in_h = r_oh * stride_h + (kh * dil_h) - pad_h
                in_w = r_ow * stride_w + (kw * dil_w) - pad_w

                # check boundary
                valid_h = (0 <= in_h) & (in_h < IN_H)
                valid_w = (0 <= in_w) & (in_w < IN_W)

                # broadcast valid
                valid = valid_h[:, None] & valid_w[None, :]
                # gather input
                in_offset_base = b * (IN_C * IN_H * IN_W) + ic * (IN_H * IN_W)
                for i in range(BLOCK_M):
                    for j in range(BLOCK_M):
                        if valid[i, j]:
                            inp_val = tl.load(input_ptr + in_offset_base + in_h[i]*IN_W + in_w[j])
                        else:
                            inp_val = 0.0
                        acc[i, j] += inp_val * w_val

    # Add bias if available
    if tl.static_assert(bias_ptr != 0, ""):  # 0 is pointer null check in Triton
        bias_val = tl.load(bias_ptr + oc_block)
        for i in range(BLOCK_M):
            for j in range(BLOCK_M):
                acc[i, j] += bias_val

    # Apply sigmoid
    for i in range(BLOCK_M):
        for j in range(BLOCK_M):
            acc[i, j] = 1.0 / (1.0 + tl.exp(-acc[i, j]))

    # store to output
    out_offset_base = b * (OUT_C * out_H * out_W) + oc_block * (out_H * out_W)
    for i in range(BLOCK_M):
        oh_pos = r_oh[i]
        if oh_pos < out_H:
            for j in range(BLOCK_M):
                ow_pos = r_ow[j]
                if ow_pos < out_W:
                    tl.store(output_ptr + out_offset_base + oh_pos * out_W + ow_pos, acc[i, j])


def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    # Ensure CUDA
    assert input.is_cuda, "input must be on CUDA."
    assert weight.is_cuda, "weight must be on CUDA."
    if bias is not None:
        assert bias.is_cuda, "bias must be on CUDA."

    # Parse stride
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    # Parse padding
    if isinstance(padding, int):
        pad_h, pad_w = padding, padding
    elif isinstance(padding, tuple):
        pad_h, pad_w = padding
    else:
        # simplistic assumption if 'valid' or 'same' is passed
        if padding == 'valid':
            pad_h, pad_w = 0, 0
        elif padding == 'same':
            # 'same' for stride=1 only
            # compute kH,kW from weight
            _, _, kH, kW = weight.shape
            pad_h = (kH - 1) // 2
            pad_w = (kW - 1) // 2
        else:
            raise ValueError("Unsupported padding option {}".format(padding))

    # Parse dilation
    if isinstance(dilation, int):
        dil_h, dil_w = dilation, dilation
    else:
        dil_h, dil_w = dilation

    B, IN_C, IN_H, IN_W = input.shape
    OUT_C, w_ic, K_H, K_W = weight.shape
    assert w_ic * groups == IN_C, "weight shape and groups do not match input channels"

    # Output shape
    out_H = (IN_H + 2*pad_h - dil_h*(K_H - 1) - 1)//stride_h + 1
    out_W = (IN_W + 2*pad_w - dil_w*(K_W - 1) - 1)//stride_w + 1
    # Allocate 'out' if needed
    if out is None:
        out = torch.empty((B, OUT_C, out_H, out_W), dtype=input.dtype, device=input.device)

    # Launch kernel
    BLOCK_M = 8  # tile dimension for OH, OW
    # We map grid:
    #   dim 0: B * OUT_C
    #   dim 1: (out_H + BLOCK_M - 1)//BLOCK_M
    #   dim 2: (out_W + BLOCK_M - 1)//BLOCK_M
    grid_0 = B * OUT_C
    grid_1 = (out_H + BLOCK_M - 1) // BLOCK_M
    grid_2 = (out_W + BLOCK_M - 1) // BLOCK_M

    bias_ptr = bias if bias is not None else torch.zeros((OUT_C,), device=input.device, dtype=input.dtype)
    # Triton does not allow None pointers at kernel time, so pass a zeroed tensor if bias=None

    _sigmoid_conv2d_kernel[(grid_0, grid_1, grid_2)](
        input_ptr=input, 
        weight_ptr=weight, 
        bias_ptr=bias_ptr, 
        output_ptr=out, 
        B=B, IN_C=IN_C, IN_H=IN_H, IN_W=IN_W,
        OUT_C=OUT_C, K_H=K_H, K_W=K_W,
        stride_h=stride_h, stride_w=stride_w,
        pad_h=pad_h, pad_w=pad_w,
        dil_h=dil_h, dil_w=dil_w,
        groups=groups,
        out_H=out_H, out_W=out_W,
        BLOCK_M=BLOCK_M, BLOCK_N=1  # Unused second block dimension
    )
    return out
