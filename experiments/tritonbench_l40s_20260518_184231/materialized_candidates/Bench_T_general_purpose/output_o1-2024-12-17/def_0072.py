import triton
import triton.language as tl
import math

@triton.jit
def _relu_batch_norm_conv2d_kernel(
    input_ptr,         # [B, IN_C, IN_H, IN_W]
    weight_ptr,        # [OUT_C, IN_C, KH, KW]
    bias_ptr,          # [OUT_C] or None
    out_ptr,           # [B, OUT_C, OUT_H, OUT_W]
    run_mean_ptr,      # [OUT_C] or None
    run_var_ptr,       # [OUT_C] or None
    bn_weight_ptr,     # [OUT_C] or None (gamma)
    bn_bias_ptr,       # [OUT_C] or None (beta)
    B, IN_C, IN_H, IN_W,
    OUT_C, KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    DIL_H, DIL_W,
    EPS,               # float
    # We do not update running stats here, so training/momentum not applied in the kernel
    # For simplicity, ignoring groups in this example
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one output element or a small tile. Here each program does 1 element.
    pid = tl.program_id(0)
    # Decompose pid into (b, oc, oh, ow):
    # out shape = [B, OUT_C, OUT_H, OUT_W] => total elements = B * OUT_C * OUT_H * OUT_W
    # We'll compute OUT_H and OUT_W:
    OUT_H = (IN_H + 2 * PAD_H - DIL_H * (KH - 1) - 1) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - DIL_W * (KW - 1) - 1) // STRIDE_W + 1

    # Flattened idx -> a 4D index
    total_elems = B * OUT_C * OUT_H * OUT_W
    if pid >= total_elems:
        return

    ow = pid % OUT_W
    oh = (pid // OUT_W) % OUT_H
    oc = (pid // (OUT_W * OUT_H)) % OUT_C
    b  = pid // (OUT_W * OUT_H * OUT_C)

    # Convolution
    conv_val = 0.0
    # For each ic in [0..IN_C):
    for ic in range(IN_C):
        for kh in range(KH):
            for kw in range(KW):
                in_h = oh * STRIDE_H + kh * DIL_H - PAD_H
                in_w = ow * STRIDE_W + kw * DIL_W - PAD_W
                if (0 <= in_h < IN_H) and (0 <= in_w < IN_W):
                    inp_idx = b * IN_C * IN_H * IN_W + ic * IN_H * IN_W + in_h * IN_W + in_w
                    wgt_idx = oc * IN_C * KH * KW + ic * KH * KW + kh * KW + kw
                    inp_val = tl.load(input_ptr + inp_idx)
                    wgt_val = tl.load(weight_ptr + wgt_idx)
                    conv_val += inp_val * wgt_val

    # Add bias if given
    if bias_ptr != 0:
        bias_val = tl.load(bias_ptr + oc)
        conv_val += bias_val

    # BatchNorm
    # For simplicity, we assume run_mean_ptr / run_var_ptr are not None
    mean_val = tl.load(run_mean_ptr + oc) if run_mean_ptr != 0 else 0.0
    var_val  = tl.load(run_var_ptr  + oc) if run_var_ptr  != 0 else 1.0
    gamma    = tl.load(bn_weight_ptr + oc) if bn_weight_ptr != 0 else 1.0
    beta     = tl.load(bn_bias_ptr  + oc) if bn_bias_ptr  != 0 else 0.0

    normed = (conv_val - mean_val) / tl.sqrt(var_val + EPS)
    bn_out = normed * gamma + beta

    # ReLU
    relu_out = tl.max(bn_out, 0.0)

    # Store
    out_idx = b * OUT_C * OUT_H * OUT_W + oc * OUT_H * OUT_W + oh * OUT_W + ow
    tl.store(out_ptr + out_idx, relu_out)


def relu_batch_norm_conv2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    running_mean=None,
    running_var=None,
    bn_weight=None,
    bn_bias=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
    inplace=False
):
    """
    Applies 2D convolution over the input tensor, followed by batch normalization
    and then applies the ReLU activation function in one combined step.

    Args:
        input (Tensor): (B, IN_C, IN_H, IN_W)
        weight (Tensor): (OUT_C, IN_C/groups, KH, KW)
        bias (Tensor, optional): (OUT_C)
        stride (int or tuple, optional): Default: 1
        padding (int, tuple, or string, optional): Default: 0
        dilation (int or tuple, optional): Default: 1
        groups (int, optional): Default: 1
        running_mean (Tensor, optional): (OUT_C)
        running_var (Tensor, optional): (OUT_C)
        bn_weight (Tensor, optional): (OUT_C) gamma
        bn_bias (Tensor, optional): (OUT_C) beta
        training (bool, optional): If True, updates running stats (not fully supported in kernel). Default: False
        momentum (float, optional): Default: 0.1
        eps (float, optional): Default: 1e-5
        inplace (bool, optional): If True, ReLU is done in-place (not used here). Default: False

    Returns:
        Tensor: Output tensor of shape (B, OUT_C, OUT_H, OUT_W)
    """

    # Extract shapes
    B, IN_C, IN_H, IN_W = input.shape
    OUT_C, _, KH, KW = weight.shape

    # Handle stride, padding, dilation as tuples
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h, pad_w = padding, padding
    else:
        pad_h, pad_w = padding

    if isinstance(dilation, int):
        dil_h, dil_w = dilation, dilation
    else:
        dil_h, dil_w = dilation

    # Compute output spatial dims
    out_h = (IN_H + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    out_w = (IN_W + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

    import torch
    # Allocate output
    output = torch.empty((B, OUT_C, out_h, out_w), dtype=input.dtype, device=input.device)

    # Convert any None bias/BN params to 0 for passing as pointer (triton can't handle None)
    bias_ptr = bias.data_ptr() if bias is not None else 0
    run_mean_ptr = running_mean.data_ptr() if running_mean is not None else 0
    run_var_ptr = running_var.data_ptr() if running_var is not None else 0
    bn_weight_ptr = bn_weight.data_ptr() if bn
