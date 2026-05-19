import triton
import triton.language as tl

@triton.jit
def _conv2d_add_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr, out_ptr,
    N, C, H, W, OC, KH, KW,
    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups,
    alpha, has_bias, has_other, other_is_scalar,
    stridei0, stridei1, stridei2, stridei3,
    stridew0, stridew1, stridew2, stridew3,
    strideo0, strideo1, strideo2, strideo3,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Compute output coordinates
    n = tl.program_id(0)
    oc = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    ohow = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    # Each ohow corresponds to a 2D (oh, ow) coordinate
    oh = ohow // (W // stride_w if stride_w != 0 else W)
    ow = ohow % (W // stride_w if stride_w != 0 else W)

    # Deduce the actual output height/width from convolution shape
    out_H = (H + 2*pad_h - dil_h*(KH-1) - 1) // stride_h + 1
    out_W = (W + 2*pad_w - dil_w*(KW-1) - 1) // stride_w + 1

    # Check bounds
    in_range_oh = (oh < out_H)
    in_range_ow = (ow < out_W)
    in_range_oc = (oc < OC)

    # Prepare accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Group size for in_channels
    cin_per_group = C // groups
    cout_per_group = OC // groups

    # For each valid oc
    for ioc in range(BLOCK_M):
        if not in_range_oc[ioc]:
            continue

    # Loop over input channels for the group
    for g in range(groups):
        oc_start = g * cout_per_group
        ic_start = g * cin_per_group
        oc_end = oc_start + cout_per_group

        # Each oc in block
        for ioc in range(BLOCK_M):
            c_out = oc[ioc]
            if c_out < oc_start or c_out >= oc_end:
                continue

            # Compute local index within group
            c_out_local = c_out - oc_start

            # Accumulate convolution
            for ic in range(cin_per_group):
                for kh in range(KH):
                    for kw in range(KW):
                        # Compute input h/w
                        oh_i = oh * stride_h + kh * dil_h - pad_h
                        ow_i = ow * stride_w + kw * dil_w - pad_w

                        # Check input boundaries
                        valid_h = (oh_i >= 0) & (oh_i < H)
                        valid_w = (ow_i >= 0) & (ow_i < W)
                        valid_out = in_range_oh & in_range_ow & valid_h & valid_w

                        # Indices for input/weight
                        in_idx = (n * stridei0
                                  + (ic_start + ic) * stridei1
                                  + oh_i * stridei2
                                  + ow_i * stridei3)
                        w_idx = (c_out * stridew0
                                 + ic * stridew1
                                 + kh * stridew2
                                 + kw * stridew3)

                        # Load if valid
                        inp_val = tl.where(valid_out,
                                           tl.load(input_ptr + in_idx, mask=valid_out, other=0.0),
                                           0.0)
                        w_val = tl.load(weight_ptr + w_idx)

                        # Accumulate
                        acc[ioc, :] += inp_val * w_val

    # Add bias if present
    if has_bias:
        for ioc in range(BLOCK_M):
            c_out = oc[ioc]
            if in_range_oc[ioc]:
                b_val = tl.load(bias_ptr + c_out)
                acc[ioc, :] += b_val

    # Add alpha * other if provided
    if has_other:
        for ioc in range(BLOCK_M):
            c_out = oc[ioc]
            if in_range_oc[ioc]:
                if other_is_scalar:
                    acc[ioc, :] += alpha * tl.load(other_ptr)
                else:
                    # Compute same oh, ow indices
                    ohow_flat = ohow
                    if in_range_oh & in_range_ow:
                        other_index = ohow_flat
                        # If other has shape matching out, stride depends on out's shape
                        other_val = tl.load(other_ptr + c_out*strideo1 + other_index*strideo2, other=0.0)
                        acc[ioc, :] += alpha * other_val

    # Store result
    for ioc in range(BLOCK_M):
        c_out = oc[ioc]
        if in_range_oc[ioc]:
            ohow_flat = ohow
            valid_pos = in_range_oh & in_range_ow
            out_index = (n * strideo0 + c_out * strideo1 + ohow_flat * strideo2)
            tl.store(out_ptr + out_index, acc[ioc, :], mask=valid_pos)

def conv2d_add(input, weight, bias=None, other=None,
               stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    """
    conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None) -> Tensor
    Applies a 2D convolution over an input image, optionally adds a bias, then adds alpha * other to the result.
    """
    import math
    import torch

    # Check groups
    if input.shape[1] % groups != 0 or weight.shape[0] % groups != 0:
        raise ValueError("groups must divide both in_channels and out_channels")

    # Convert stride, padding, dilation to tuples if needed
    def to_2tuple(x):
        return x if isinstance(x, tuple) else (x, x)

    stride_h, stride_w = to_2tuple(stride)
    if isinstance(padding, str):
        if padding.lower() == 'same':
            # 'same' padding for conv
            padding_h = math.ceil(((input.shape[2] - 1) * (dilation if isinstance(dilation, int) else dilation[0]) + (weight.shape[2] - 1) + 1 - input.shape[2]) / 2)
            padding_w = math.ceil(((input.shape[3] - 1) * (dilation if isinstance(dilation, int) else dilation[1]) + (weight.shape[3] - 1) + 1 - input.shape[3]) / 2)
        elif padding.lower() == 'valid':
            padding_h, padding_w = 0, 0
        else:
            raise ValueError("Unsupported padding string. Use 'same' or 'valid'.")
    else:
        padding_h, padding_w = to_2tuple(padding)

    dil_h, dil_w = to_2tuple(dilation)

    # Input dimensions
    N, C, H, W = input.shape
    OC, _, KH, KW = weight.shape

    # Output shape
    out_h = (H + 2*padding_h - dil_h*(KH-1) - 1)//stride_h + 1
    out_w = (W + 2*padding_w - dil_w*(KW-1) - 1)//stride_w + 1

    if out is None:
        out = input.new_empty((N, OC, out_h, out_w))

    # Create pointers
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    out_ptr = out.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else 0
    has_bias = 1 if bias is not None else 0

    # other can be None, scalar, or Tensor
    if other is None:
        other_ptr = 0
        has_other = 0
        other_is_scalar = 0
    elif isinstance(other, (int, float)):
        # Create a temp tensor for scalar
        other_t = input.new_tensor([other])
        other_ptr = other_t.data_ptr()
        has_other = 1
        other_is_scalar = 1
    else:
        other_ptr = other.data_ptr()
        has_other = 1
        other_is_scalar = 0

    # Strides
    stridei0, stridei1, stridei2, stridei3 = input.stride()
    stridew0, stridew1, stridew2, stridew3 = weight.stride()
    strideo0, strideo1, strideo2, strideo3 = out.stride()

    # Tune block sizes
    BLOCK_M = 1   # for out_channels
    BLOCK_N = 128 # for ohow dimension

    grid = (N, math.ceil(OC / BLOCK_M), math.ceil((out_h * out_w) / BLOCK_N))

    _conv2d_add_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, other_ptr, out_ptr,
        N, C, H, W, OC, KH, KW,
        stride_h, stride_w, padding_h, padding_w, dil_h, dil_w, groups,
        alpha, has_bias, has_other, other_is_scalar,
        stridei0, stridei1, stridei2, stridei3,
        stridew0, stridew1, stridew2, stridew3,
        strideo0, strideo1, strideo2, strideo3,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )

    return out
