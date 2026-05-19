import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_fused(
    X,  # input tensor
    Y,  # output tensor
    W,  # learned weights
    stride_xm,  # stride in the input tensor
    stride_ym,  # stride in the output tensor
    M,  # number of rows
    N,  # number of columns (features)
    eps: tl.float32,  # small epsilon value to avoid division by zero
    BLOCK_SIZE: tl.constexpr,  # block size for parallel processing
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    # Load the row from the input tensor
    row_offsets = offsets * stride_xm
    x_ptrs = X + row_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Compute the mean of the row
    mean = tl.sum(x, axis=0) / N

    # Compute the variance of the row
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N

    # Compute the reciprocal of the standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)

    # Apply the learned weights to the normalized data
    w_ptr = W + offsets
    w = tl.load(w_ptr, mask=mask, other=1.0)
    y = x_centered * rstd * w

    # Store the result in the output tensor
    y_ptrs = Y + row_offsets
    tl.store(y_ptrs, y, mask=mask)

import torch
import triton
import triton.language as tl

class TritonLlamaRMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(TritonLlamaRMSNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        assert x.is_cuda, "Input tensor must be on a CUDA device"
        assert x.dim() == 2, "Input tensor must be 2D (batch_size, features)"
        assert x.size(1) < 64 * 1024, "Feature dimension must be less than 64KB"

        M, N = x.size()
        BLOCK_SIZE = 128  # Adjust this block size as needed

        # Allocate output tensor
        y = torch.empty_like(x)

        # Launch the Triton kernel
        grid = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
        rms_norm_fwd_fused[grid, BLOCK_SIZE](
            x, y, self.weight, x.stride(0), y.stride(0), M, N, self.eps, BLOCK_SIZE
        )

        return y
