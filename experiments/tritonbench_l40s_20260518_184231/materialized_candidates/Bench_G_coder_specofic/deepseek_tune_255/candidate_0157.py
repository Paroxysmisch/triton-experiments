import triton
import triton.language as tl
import torch
from torch import Tensor
from torch.nn import Module

@triton.jit
def rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y, mask=mask)

class TritonLlamaRMSNorm(Module):
    def __init__(self, weight: Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x: Tensor):
        # reshape input data into 2D tensor
        x_arg = x.view(-1, x.shape[-1])
        # construct output tensor
        y = torch.empty_like(x_arg)
        # get row stride
        x_stride = x_arg.stride()[0]
        # get number of columns
        n = x_arg.size(1)
        # get block size
        BLOCK_SIZE = max(triton.next_power_of_2(n), 16)
        # enqueue kernel
        rms_norm_fwd_fused[(x_arg.shape[0],)](
            x_arg,
            y,
            self.weight,
            x_stride,
            n,
            self.variance_epsilon,
            BLOCK_SIZE,
        )
        return y
