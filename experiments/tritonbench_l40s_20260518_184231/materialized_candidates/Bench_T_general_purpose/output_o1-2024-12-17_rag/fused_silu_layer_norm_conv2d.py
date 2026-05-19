import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.runtime.jit import get_cuda_stream


@triton.jit
def _layernorm_silu_kernel(
    input_ptr,        # pointer to input tensor (N, C, H, W)
    output_ptr,       # pointer to output tensor (N, C, H, W)
    gamma_ptr,        # pointer to LN scale (C,)
    eps,              # epsilon for LN
    stride_n,         # stride for N dimension
    stride_c,         # stride for C dimension
    stride_h,         # stride for H dimension
    stride_w,         # stride for W dimension
    batch_size,       # N
    n_channels,       # C
    height,           # H
    width,            # W
    BLOCK_SIZE: tl.constexpr
):
    """
    Each program handles one (n, h, w) coordinate. We compute the mean/var across C
    for that coordinate, then apply LayerNorm scaling (gamma) and SiLU activation.
    """
    pid = tl.program_id(0)
    # Decompose pid into n, h, w
    w_idx = pid % width
    tmp = pid // width
    h_idx = tmp % height
    n_idx = tmp // height

    # Accumulate sum_x and sum_x2 across all channels
    sum_x = tl.float32(0.0)
    sum_x2 = tl.float32(0.0)

    c_block = 0
    # First pass to compute mean/variance
    while c_block < n_channels:
        offsets = tl.arange(0, BLOCK_SIZE)
        c_idx = c_block + offsets
        mask = c_idx < n_channels

        # location in memory
        in_ptr = input_ptr + (n_idx * stride_n) + (h_idx * stride_h) + (w_idx * stride_w)
        in_ptr += c_idx * stride_c
        x = tl.load(in_ptr, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        sum_x += tl.sum(x_f32, mask=mask)
        sum_x2 += tl.sum(x_f32 * x_f32, mask=mask)
        c_block += BLOCK_SIZE

    mean = sum_x / n_channels
    var = (sum_x2 / n_channels) - (mean * mean)

    # Second pass: apply LayerNorm and SiLU
    c_block = 0
    while c_block < n_channels:
        offsets = tl.arange(0, BLOCK_SIZE)
        c_idx = c_block + offsets
        mask = c_idx < n_channels

        in_ptr = input_ptr + (n_idx * stride_n) + (h_idx * stride_h) + (w_idx * stride_w)
        in_ptr += c_idx * stride_c
        out_ptr = output_ptr + (n_idx * stride_n) + (h_idx * stride_h) + (w_idx * stride_w)
        out_ptr += c_idx * stride_c
