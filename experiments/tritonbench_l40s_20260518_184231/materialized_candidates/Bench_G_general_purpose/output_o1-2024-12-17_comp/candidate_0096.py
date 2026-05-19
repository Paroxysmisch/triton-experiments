import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_fused(
    x_ptr, w_ptr, y_ptr,
    M, N,
    stride_x_row, stride_x_col,
    stride_w,
    stride_y_row, stride_y_col,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    # Guard against out-of-range program ids
    if row_id >= M:
        return

    # ---------------------
    # 1) Compute sum of squares
    # ---------------------
    sum_squares = tl.zeros([1], dtype=tl.float32)
    block_offsets = tl.arange(0, BLOCK_SIZE)
    # Loop over all columns in BLOCK_SIZE increments
    for col_start in range(0, N, BLOCK_SIZE):
        col_ids = col_start + block_offsets
        mask = col_ids < N
        x = tl.load(
            x_ptr + row_id * stride_x_row + col_ids * stride_x_col,
            mask=mask,
            other=0.0
        )
        sum_squares += tl.sum(x * x, axis=0)

    # Compute rstd = 1.0 / sqrt(mean)
    mean = sum_squares.to(tl.float32) / tl.float32(N)
    rstd = 1.0 / tl.sqrt(mean)

    # ---------------------
    # 2) Apply normalization and weight
    # ---------------------
    for col_start in range(0, N, BLOCK_SIZE):
        col_ids = col_start + block_offsets
        mask = col_ids < N
        x = tl.load(
            x_ptr + row_id * stride_x_row + col_ids * stride_x_col,
            mask=mask,
            other=0.0
        )
        w = tl.load(
            w_ptr + col_ids * stride_w,
            mask=mask,
            other=0.0
        )
        y = x * rstd * w
        tl.store(
            y_ptr + row_id * stride_y_row + col_ids * stride_y_col,
            y,
            mask=mask
        )

def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor):
    # x shape: [*] x N
    # weight shape: [N]
    # Output: y same shape as x
    # Flatten all leading dimensions
    x_shape = x.shape
    N = x_shape[-1]
    M = x.numel() // N

    # Make contiguous if necessary
    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    y = torch.empty_like(x_contig)

    # Strides
    stride_x_row = x_contig.stride(0) * x_contig.shape[0] // M if len(x_shape) > 1 else 0
    stride_x_col = x_contig.stride(-1)
    stride_w = weight_contig.stride(0)
    stride_y_row = y.stride(0) * y.shape[0] // M if len(x_shape) > 1 else 0
    stride_y_col = y.stride(-1)

    # Heuristics for block size and warps
    BLOCK_SIZE = 128
    # Number of warps based on feature size
    num_warps = 4 if N >= 2048 else 1

    grid = (M,)
    _rms_norm_fwd_fused[grid](
        x_contig, weight_contig, y,
        M, N,
        stride_x_row, stride_x_col,
        stride_w,
        stride_y_row, stride_y_col,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    # Reshape back to original shape
    return y.view(*x_shape)
