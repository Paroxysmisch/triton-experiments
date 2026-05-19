import torch
import triton
import triton.language as tl

@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))

@triton.jit
def _calc_mean_and_inv_std_kernel(
    INPUT, OUTPUT_MEAN, OUTPUT_INV_STD,
    BATCH, CHAN, EPS,
    stride_input_batch, stride_input_chan
):
    pid = tl.program_id(0)
    if pid >= CHAN:
        return

    # Each program handles one channel across the batch dimension.
    channel_offset = pid
    mean_acc = tl.zeros([1], dtype=tl.float32)
    var_acc = tl.zeros([1], dtype=tl.float32)

    # Accumulate mean/variance across batch
    for i in range(BATCH):
        inp_val = tl.load(INPUT + i * stride_input_batch + channel_offset * stride_input_chan)
        mean_acc += inp_val
    mean_val = mean_acc / BATCH

    for i in range(BATCH):
        inp_val = tl.load(INPUT + i * stride_input_batch + channel_offset * stride_input_chan)
        diff = inp_val - mean_val
        var_acc += diff * diff
    var_val = var_acc / BATCH
    inv_std_val = 1.0 / tl.sqrt(var_val + EPS)

    # Store
    tl.store(OUTPUT_MEAN + channel_offset, mean_val)
    tl.store(OUTPUT_INV_STD + channel_offset, inv_std_val)

@triton.jit
def _apply_bn_sigmoid_kernel(
    INPUT, OUTPUT,
    MEAN, INV_STD,
    WEIGHT, BIAS,
    BATCH, CHAN,
    stride_input_batch, stride_input_chan,
    stride_output_batch, stride_output_chan,
    apply_weight_bias: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= CHAN:
        return

    channel_offset = pid
    mean_val = tl.load(MEAN + channel_offset)
    inv_std_val = tl.load(INV_STD + channel_offset)
    if apply_weight_bias:
        w = tl.load(WEIGHT + channel_offset)
        b = tl.load(BIAS + channel_offset)
    else:
        w = 1.0
        b = 0.0

    for i in range(BATCH):
        inp_val = tl.load(INPUT + i * stride_input_batch + channel_offset * stride_input_chan)
        norm_val = (inp_val - mean_val) * inv_std_val
        val = norm_val * w + b
        # apply sigmoid
        out_val = 1.0 / (1.0 + tl.exp(-val))
        tl.store(OUTPUT + i * stride_output_batch + channel_offset * stride_output_chan, out_val)

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None,
                       training=False, momentum=0.1, eps=1e-5):
    """
    Applies Batch Normalization to 'input' across its channels,
    then applies a sigmoid activation to the normalized result.

    Args:
        input (Tensor): Shape (N, C) or (N, C, L).
        running_mean (Tensor): Running mean, shape (C,).
        running_var (Tensor): Running variance, shape (C,).
        weight (Tensor, optional): Channel scale factors (C,). Default: None.
        bias (Tensor, optional): Channel shift factors (C,).  Default: None.
        training (bool, optional): If True, uses batch statistics to update running stats. Default: False.
        momentum (float, optional): Momentum for updating running stats. Default: 0.1.
        eps (float, optional): Small epsilon for numerical stability. Default: 1e-5.

    Returns:
        Tensor: Normalized, sigmoid-activated tensor of the same shape as input.
    """
    # Ensure input is 2D or 3D
    if input.dim() == 2:
        # (N, C)
        N, C = input.shape
        L = 1
        reshaped_input = input
    elif input.dim() == 3:
        # (N, C, L)
        N, C, L = input.shape
        reshaped_input = input.view(N, C * L)
    else:
        raise ValueError("Input must be 2D or 3D.")

    device = input.device
    dtype = input.dtype

    # We will compute stats across the first dimension (batch-like dimension) for each channel.
    if L > 1:
        # transform so shape = (N*L, C) for easier channel-wise iteration
        reshaped_input = reshaped_input.transpose(0, 1)  # shape (C, N*L)
        B = N * L
        chan_size = C
        stride_input_batch = reshaped_input.stride(0)
        stride_input_chan = reshaped_input.stride(1)
    else:
        # shape = (N, C) => transpose => shape (C, N)
        reshaped_input = reshaped_input.transpose(0, 1)  # shape (C, N)
        B = N
        chan_size = C
        stride_input_batch = reshaped_input.stride(0)
        stride_input_chan = reshaped_input.stride(1)

    out_tensor = torch.empty_like(reshaped_input)

    # Allocate intermediate mean/inv_std on device
    cur_mean = torch.empty(chan_size, dtype=torch.float32, device=device)
    cur_inv_std = torch.empty(chan_size, dtype=torch.float32, device=device)

    grid = (chan_size,)

    # Compute per-channel mean & inv_std for this batch
    _calc_mean_and_inv_std_kernel[grid](
        reshaped_input,        # INPUT
        cur_mean,              # OUTPUT_MEAN
        cur_inv_std,           # OUTPUT_INV_STD
        B,                     # BATCH
        chan_size,            # CHAN
        eps,                   # EPS
        stride_input_batch,    # stride_input_batch
        stride_input_chan      # stride_input_chan
    )

    if training:
        # Update running mean/var
        # var = 1.0 / (inv_std^2), so let's get var
        cur_var = 1.0 / (cur_inv_std ** 2) - eps
        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(cur_mean * momentum)
            running_var.mul_(1 - momentum).add_(cur_var * momentum)

    # If not training, use running stats
    if not training:
        cur_mean = running_mean
        cur_var = running_var
        cur_inv_std = 1.0 / torch.sqrt(cur_var + eps)

    apply_weight_bias = (weight is not None) and (bias is not None)
    if weight is None:
        weight = torch.ones_like(cur_mean, dtype=torch.float32, device=device)
    if bias is None:
        bias = torch.zeros_like(cur_mean, dtype=torch.float32, device=device)

    _apply_bn_sigmoid_kernel[grid](
        reshaped_input,                  # INPUT
        out_tensor,                      # OUTPUT
        cur_mean,                        # MEAN
        cur_inv_std,                     # INV_STD
        weight,                          # WEIGHT
        bias,                            # BIAS
        B,                               # BATCH
        chan_size,                       # CHAN
        stride_input_batch,              # stride_input_batch
        stride_input_chan,               # stride_input_chan
        out_tensor.stride(0),           # stride_output_batch
        out_tensor.stride(1),           # stride_output_chan
        apply_weight_bias=apply_weight_bias
    )

    # Reshape/output back to original shape
    if L > 1:
        # shape was (C, N*L), revert to (N, C, L)
        out_tensor = out_tensor.transpose(0, 1).contiguous().view(N, C, L)
    else:
        # shape was (C, N), revert to (N, C)
        out_tensor = out_tensor.transpose(0, 1).contiguous()

    return out_tensor.to(dtype)
