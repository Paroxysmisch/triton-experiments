import math
import torch
import triton
import triton.language as tl

@triton.jit
def cos_avg_pool1d_kernel(
    input_ptr, output_ptr,
    B, C, IN_W, OUT_W,
    kernel_size, stride, padding,
    count_include_pad,
    BLOCK_W: tl.constexpr
):
    batch_channel_idx = tl.program_id(0)
    out_col_start = tl.program_id(1) * BLOCK_W
    col_range = out_col_start + tl.arange(0, BLOCK_W)

    # Compute which batch/channel we are in
    # Each program id(0) corresponds to a unique (batch, channel) pair
    # B*C = total number of (batch, channel) in the input
    # batch_channel_idx in [0, B*C)
    # We'll flatten them as one dimension
    # offset for reading input, writing output
    in_offset = batch_channel_idx * IN_W
    out_offset = batch_channel_idx * OUT_W

    # Initialize accumulators
    sum_values = tl.zeros([BLOCK_W], dtype=tl.float32)
    valid_counts = tl.zeros([BLOCK_W], dtype=tl.int32)  # only used if not count_include_pad

    # Loop over the pooling window
    for k in range(kernel_size):
        # For each output element, compute the corresponding input index
        in_idx = col_range * stride - padding + k
        # Check bounds
        in_bounds = (in_idx >= 0) & (in_idx < IN_W) & (col_range < OUT_W)
        # If in bounds, read, take cos, accumulate
        input_val = tl.where(
            in_bounds,
            tl.load(input_ptr + in_offset + in_idx, mask=in_bounds, other=0.0),
            0.0
        )
        cos_val = tl.cos(input_val)
        sum_values += cos_val

        if not count_include_pad:
            # Count only valid input positions
            valid_counts += in_bounds.to(tl.int32)

    # Compute the divisor
    if count_include_pad:
        divisor = kernel_size
    else:
        # Avoid division by zero
        divisor = tl.where(valid_counts == 0, 1, valid_counts)
    result = sum_values / divisor

    # Write back results for valid col_range
    out_mask = col_range < OUT_W
    tl.store(output_ptr + out_offset + col_range, result, mask=out_mask)


def cos_avg_pool1d(
    input: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True
) -> torch.Tensor:
    if stride is None:
        stride = kernel_size

    # input shape: (B, C, W)
    B, C, IN_W = input.shape

    # Compute output width
    if ceil_mode:
        OUT_W = math.ceil((IN_W + 2 * padding - kernel_size) / stride + 1)
    else:
        OUT_W = math.floor((IN_W + 2 * padding - kernel_size) / stride + 1)
    OUT_W = max(OUT_W, 0)

    # Allocate output
    out = torch.empty((B, C, OUT_W), device=input.device, dtype=input.dtype)

    # Flatten batch and channels as one dimension for the kernel
    BC = B * C

    # Launch the Triton kernel
    BLOCK_W = 128
    grid = (BC, (OUT_W + BLOCK_W - 1) // BLOCK_W)

    cos_avg_pool1d_kernel[grid](
        input, out,
        B, C, IN_W, OUT_W,
        kernel_size, stride, padding,
        count_include_pad,
        BLOCK_W=BLOCK_W
    )

    return out
