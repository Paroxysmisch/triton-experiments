import triton
import triton.language as tl

@triton.jit
def _relu_conv2d_kernel(
    input_ptr,           # *f32
    weight_ptr,          # *f32
    bias_ptr,            # *f32 (optional)
    output_ptr,          # *f32
    N, C_in, H_in, W_in,
    C_out, groups,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    in_sN, in_sC, in_sH, in_sW,
    wt_sCo, wt_sCi, wt_sKh, wt_sKw,
    out_sN, out_sC, out_sH, out_sW,
    inplace,  # 0 or 1
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Coordinates in output
    out_h = pid_hw // BLOCK_N
    out_w = pid_hw % BLOCK_N
    out_c = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)

    # Check if within output bounds
    out_h_in_range = out_h < ((H_in + 2*pad_h - dil_h*(kernel_h-1) - 1) // stride_h + 1)
    out_w_in_range = out_w < ((W_in + 2*pad_w - dil_w*(kernel_w-1) - 1) // stride_w + 1)
    out_c_in_range = out_c < C_out

    # If out of valid output range, exit early
    if not (out_h_in_range and out_w_in_range):
        return

    # Compute input spatial start
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w

    # Each output pixel: sum over in_channels/groups * kernel height * kernel width
    # In general: out_c ranges 0..C_out, but we have groups factor
    # group size for channels
    group_size_in = C_in // groups
    group_id = out_c // (C_out // groups)
    c_in_offset = group_id * group_size_in

    # Accumulator
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over kernel area
    for kh in range(kernel_h):
        in_h = in_h_start + kh * dil_h
        if (in_h < 0) or (in_h >= H_in):
            continue
        for kw in range(kernel_w):
            in_w = in_w_start + kw * dil_w
            if (in_w < 0) or (in_w >= W_in):
                continue
            # Loop over group_size_in
            for gc in range(group_size_in):
                # Input channel index
                ci = c_in_offset + gc
                # Load input
                inp_ofs = pid_n * in_sN + ci * in_sC + in_h * in_sH + in_w * in_sW
                val_inp = tl.load(input_ptr + inp_ofs)
                # Load weight
                w_ofs = (out_c * wt_sCo) + gc * wt_sCi + kh * wt_sKh + kw * wt_sKw
                w_val = tl.load(weight_ptr + w_ofs, mask=out_c_in_range, other=0.0)
                # FMA
                acc += val_inp * w_val

    # Add bias if provided
    if bias_ptr != tl.zeros((1,), dtype=tl.int1)[0]:
        b_val = tl.load(bias_ptr + out_c, mask=out_c_in_range, other=0.0)
        acc += b_val

    # Apply ReLU
    acc = tl.where(acc > 0, acc, 0.0)

    # Decide on output location
    if inplace == 1:
        # In-place: store back to input pointer
        out_ofs = pid_n * in_sN + out_c * in_sC + out_h * in_sH + out_w * in_sW
        tl.store(input_ptr + out_ofs, acc, mask=out_c_in_range & out_h_in_range & out_w_in_range)
    else:
        # Normal out
        out_ofs = pid_n * out_sN + out_c * out_sC + out_h * out_sH + out_w * out_sW
        tl.store(output_ptr + out_ofs, acc, mask=out_c_in_range & out_h_in_range & out_w_in_range)


def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    """
    relu_conv2d(input, weight, bias=None, stride=1, padding=0,
                dilation=1, groups=1, inplace=False) -> Tensor

    Functional Description:
        out = ReLU(conv2d(input, weight, bias, stride, padding, dilation, groups))
    """
    import math
    import torch

    # Ensure input is on CUDA
    if not input.is_cuda:
        raise ValueError("Input must be a CUDA tensor.")
    if not weight.is_cuda:
        raise ValueError("Weight must be a CUDA tensor.")
    if bias is not None and not bias.is_cuda:
        raise ValueError("Bias must be a CUDA tensor if provided.")

    # Turn stride, padding, dilation into pairs
    def _pair(x):
        return (x, x) if isinstance(x, int) else x

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)

    N, C_in, H_in, W_in = input.shape
    C_out, c_in_g, kernel_h, kernel_w = weight.shape
    if C_in % groups != 0:
        raise ValueError("Input channels must be divisible by groups.")
    if c_in_g * groups != C_in:
        raise ValueError("Weight shape does not match input channel grouping.")

    # Calculate output size
    H_out = (H_in + 2*pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2*pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1

    # Prepare output
    if inplace:
        out = input
    else:
        out = torch.empty((N, C_out, H_out, W_out), device=input.device, dtype=input.dtype)

    # Extract strides
    in_sN, in_sC, in_sH, in_sW = input.stride()
    wt_sCo, wt_sCi, wt_sKh, wt_sKw = weight.stride()
    if not inplace:
        out_sN, out_sC, out_sH, out_sW = out.stride()
    else:
        # If in-place, we reuse input's strides for 'storing' the result at the same memory
        out_sN, out_sC, out_sH, out_sW = in_sN, in_sC, in_sH, in_sW

    # Bias pointer (or dummy)
    bias_ptr = bias.data_ptr() if bias is not None else 0

    # Kernel launch configurations
    BLOCK_M = 32  # block out-chan
    BLOCK_N = 32  # block H*W factor

    grid = (
        N,                                  # over batch dimension
        (H_out * W_out + BLOCK_N - 1) // BLOCK_N,  # tile over H_out*W_out
        (C_out + BLOCK_M - 1) // BLOCK_M    # tile over out channels
    )

    # Launch
    _relu_conv2d_kernel[grid](
        input.data_ptr(),
        weight.data_ptr(),
        bias_ptr,
        out.data_ptr() if not inplace else input.data_ptr(),
        N, C_in, H_in, W_in,
        C_out, groups,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        in_sN, in_sC, in_sH, in_sW,
        wt_sCo, wt_sCi, wt_sKh, wt_sKw,
        out_sN, out_sC, out_sH, out_sW,
        1 if inplace else 0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )

    return out if not inplace else input
