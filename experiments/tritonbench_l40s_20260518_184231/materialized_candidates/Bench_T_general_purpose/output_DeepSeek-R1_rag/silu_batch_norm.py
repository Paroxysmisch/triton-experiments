import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def silu_batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_row_stride,
    n_cols,
    eps,
    momentum,
    training: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    row_start = prog_id * input_row_stride

    sum_x = 0.0
    sum_x2 = 0.0
    for offset in range(0, N_COLS, BLOCK_N):
        col_offsets = offset + tl.arange(0, BLOCK_N)
        mask = col_offsets < N_COLS
        input_ptrs = input_ptr + row_start + col_offsets
        x = tl.load(input_ptrs, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        sum_x += tl.sum(xf, axis=0)
        sum_x2 += tl.sum(xf * xf, axis=0)

    mean = sum_x / N_COLS
    var = (sum_x2 / N_COLS) - (mean * mean)

    if training:
        current_mean = tl.load(running_mean_ptr + prog_id)
        new_mean = (1.0 - momentum) * current_mean + momentum * mean
        tl.store(running_mean_ptr + prog_id, new_mean)

        current_var = tl.load(running_var_ptr + prog_id)
        new_var = (1.0 - momentum) * current_var + momentum * var
        tl.store(running_var_ptr + prog_id, new_var)
    else:
        mean = tl.load(running_mean_ptr + prog_id)
        var = tl.load(running_var_ptr + prog_id)

    weight = 1.0
    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + prog_id)
    bias = 0.0
    if HAS_BIAS:
        bias = tl.load(bias_ptr + prog_id)

    for offset in range(0, N_COLS, BLOCK_N):
        col_offsets = offset + tl.arange(0, BLOCK_N)
        mask = col_offsets < N_COLS
        input_ptrs = input_ptr + row_start + col_offsets
        x = tl.load(input_ptrs, mask=mask, other=0.0)
        xf = x.to(tl.float32)

        x_norm = (xf - mean) / tl.sqrt(var + eps)
        x_norm = x_norm * weight + bias

        sig = 1.0 / (1.0 + tl.exp(-x_norm))
        out = x_norm * sig

        output_ptrs = output_ptr + row_start + col_offsets
        tl.store(output_ptrs, out.to(x.dtype), mask=mask)

@torch.inference_mode()
def silu_batch_norm(
    input: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Tensor = None,
    bias: Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """
    Applies Batch Normalization followed by SiLU activation using Triton.

    Args:
        input (Tensor): Input tensor of shape (N, C, ...).
        running_mean (Tensor): Running mean tensor of shape (C,).
        running_var (Tensor): Running variance tensor of shape (C,).
        weight (Tensor, optional): Weight tensor of shape (C,). Default: None.
        bias (Tensor, optional): Bias tensor of shape (C,). Default: None.
        training (bool): Whether to update running stats. Default: False.
        momentum (float): Momentum for running stats update. Default: 0.1.
        eps (float): Epsilon for numerical stability. Default: 1e-5.

    Returns:
        Tensor: Output tensor after batch norm and SiLU.
    """
    C = running_mean.size(0)
    input_reshaped = input.contiguous().view(input.size(0), C, -1).permute(1, 0, 2).reshape(C, -1)
    output = torch.empty_like(input_reshaped)
    n_cols = input_reshaped.size(1)
    input_row_stride = input_reshaped.stride(0)

    weight = torch.ones_like(running_mean) if weight is None else weight
    bias = torch.zeros_like(running_mean) if bias is None else bias

    BLOCK_N = triton.next_power_of_2(n_cols)
    grid = (C,)
    HAS_WEIGHT = weight is not None
    HAS_BIAS = bias is not None

    def _kernel_meta():
        device = input.device
        device_idx = device.index
        stream = get_cuda_stream(device_idx)
        return dict(device=device, stream=stream, device_type=device.type)

    silu_batch_norm_kernel[grid](
        input_reshaped,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        input_row_stride,
        n_cols,
        eps,
        momentum,
        training=training,
        HAS_WEIGHT=HAS_WEIGHT,
        HAS_BIAS=HAS_BIAS,
        N_COLS=n_cols,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
        **_kernel_meta(),
    )

    output = output.view(C, input.size(0), -1).permute(1, 0, 2).view(input.shape)
    return output
