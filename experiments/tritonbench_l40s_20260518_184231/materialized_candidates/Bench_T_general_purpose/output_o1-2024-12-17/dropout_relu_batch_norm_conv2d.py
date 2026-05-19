import torch
import triton
import triton.language as tl

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    BATCH, IN_C, IN_H, IN_W, OUT_C, KH, KW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Each program handles a block of the output in height/width dimensions
    pid = tl.program_id(0)
    num_w_out = (IN_W + 2*pad_w - dilation_w*(KW-1) - 1)//stride_w + 1
    out_h = pid // num_w_out
    out_w = pid %  num_w_out

    # Check bounds
    if out_h >= (IN_H + 2*pad_h - dilation_h*(KH-1) - 1)//stride_h + 1:
        return

    # Compute pointer to output
    out_idx = (out_h * num_w_out + out_w) * OUT_C
    out_ptr = output_ptr + out_idx

    # For each channel in OUT_C, accumulate the convolution
    for oc in range(OUT_C):
        out_val = 0.0
        group_idx = oc // (OUT_C // groups)
        c_in_start = group_idx * (IN_C // groups)
        c_in_end   = c_in_start + (IN_C // groups)

        # Convolution
        for c_in in range(c_in_start, c_in_end):
            for kh in range(KH):
                ih = out_h * stride_h + kh * dilation_h - pad_h
                if ih < 0 or ih >= IN_H:
                    continue
                for kw in range(KW):
                    iw = out_w * stride_w + kw * dilation_w - pad_w
                    if iw < 0 or iw >= IN_W:
                        continue
                    inp_idx = ((ih * IN_W) + iw) * IN_C + c_in
                    w_idx   = (((oc * (IN_C//groups) + (c_in - c_in_start)) * KH) + kh) * KW + kw
                    out_val += tl.load(input_ptr + inp_idx) * tl.load(weight_ptr + w_idx)

        # Add bias if provided
        if bias_ptr != tl.nullptr:
            out_val += tl.load(bias_ptr + oc)

        # Store result
        tl.store(out_ptr + oc, out_val)


@triton.jit
def _batchnorm_kernel(
    data_ptr, mean_ptr, var_ptr, out_ptr,
    N, C, H, W, eps: tl.float32
):
    idx = tl.program_id(0)
    # Compute per-element index
    hw = H * W
    n = idx // (C * hw)
    r = idx % (C * hw)
    c = r // hw
    hw_idx = r % hw

    x = tl.load(data_ptr + idx)
    mean_val = tl.load(mean_ptr + c)
    var_val = tl.load(var_ptr + c)
    # Simple BN: (x - mean) / sqrt(var + eps)
    x_norm = (x - mean_val) / tl.sqrt(var_val + eps)
    tl.store(out_ptr + idx, x_norm)


@triton.jit
def _relu_dropout_kernel(
    data_ptr, out_ptr, mask_ptr,
    P, training, inplace,
    size, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_flag = 1.0 - P
    for i in range(BLOCK_SIZE):
        idx = offsets[i]
        if idx < size:
            val = tl.load(data_ptr + idx)
            # ReLU
            val = tl.max(val, 0.0)
            if training > 0:
                # Dropout mask
                rand = tl.random.float32(idx) % 1.0
                # Keep if rand >= p
                keep = 1.0 if (rand >= P) else 0.0
                val = val * keep / mask_flag
            if inplace > 0:
                tl.store(data_ptr + idx, val)
            else:
                tl.store(out_ptr + idx, val)
            if mask_ptr != tl.nullptr and training > 0:
                tl.store(mask_ptr + idx, keep if rand >= P else 0.0)


def dropout_relu_batch_norm_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    p=0.5,
    training=True,
    inplace=False
) -> torch.Tensor:
    """
    Applies 2D convolution -> BatchNorm -> ReLU -> Dropout sequentially.
    """
    # Ensure tensor layout
    # Assumes shape (N, C_in, H, W)
    # Weight shape (C_out, C_in//groups, kH, kW)
    N, C_in, H, W = input.shape
    C_out, _, KH, KW = weight.shape

    # Convert strides/paddings/dilations to tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    # Output shape
    out_h = (H + 2*pad_h - dilation_h*(KH-1) - 1)//stride_h + 1
    out_w = (W + 2*pad_w - dilation_w*(KW-1) - 1)//stride_w + 1

    # Prepare storage for conv output
    conv_out = torch.empty((N, C_out, out_h, out_w), dtype=input.dtype, device=input.device)

    # Flatten input for kernel: (N, H, W, C_in) with contiguous memory
    inp_contig = input.permute(0, 2, 3, 1).contiguous()
    inp_ptr = inp_contig.view(-1).data_ptr()
    # Flatten weight: (C_out,
