import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_fused(
    X,  # pointer to the input tensor
    Y,  # pointer to the output tensor
    W,  # pointer to the weight tensor
    stride_x,  # stride between rows of X
    N,  # number of columns in X
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # block size for processing columns
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x
    sum_sq = 0.0

    # Compute sum of squares for the entire row
    for offset in range(0, N, BLOCK_SIZE):
        col_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_indices < N
        x = tl.load(X + row_start + col_indices, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    # Calculate RMS and reciprocal of standard deviation (rstd)
    rstd = tl.libdevice.rsqrt(sum_sq / N + eps)

    # Apply normalization and weights
    for offset in range(0, N, BLOCK_SIZE):
        col_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_indices < N
        x = tl.load(X + row_start + col_indices, mask=mask, other=0.0)
        w = tl.load(W + col_indices, mask=mask, other=0.0)
        y = x * rstd * w
        tl.store(Y + row_start + col_indices, y, mask=mask)


class TritonLlamaRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(weight))
        self.eps = eps

    def forward(self, x):
        # Reshape input to 2D tensor (M, N) and ensure contiguous memory
        x_ = x.reshape(-1, x.size(-1)).contiguous()
        M, N = x_.shape

        # Validate weight shape
        assert self.weight.shape == (N,), f"Expected weight size ({N},), got {self.weight.shape}"

        # Allocate output tensor
        y = torch.empty_like(x_)

        # Determine optimal block size (power of two, capped at 4096)
        max_block_size = 4096
        block_size = min(triton.next_power_of_2(N), max_block_size)

        # Launch Triton kernel
        grid = (M,)
        rms_norm_fwd_fused[grid](
            x_, y, self.weight,
            x_.stride(0),  # Stride between rows
            N,            # Number of columns
            self.eps,
            BLOCK_SIZE=block_size
        )

        # Reshape output to match original input shape
        return y.reshape_as(x)
