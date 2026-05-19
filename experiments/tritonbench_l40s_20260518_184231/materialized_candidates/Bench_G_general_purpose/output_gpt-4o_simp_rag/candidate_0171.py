import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    X,   # pointer to the input
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,  # block size for loading data
):
    row = tl.program_id(0)
    X += row * stride
    DY += row * stride
    DX += row * stride

    # Load slices of X and DY
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x

    # Compute variance
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Compute gradients
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        dx = (dy - x * tl.sum(dy * x, axis=0) / N) * rstd
        tl.store(DX + cols, dx, mask=mask)

import torch

def _l2_norm_bwd(x, dy, eps=1e-5):
    x = x.contiguous()
    dy = dy.contiguous()
    dx = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape

    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _l2_norm_bwd_kernel[(M,)](
        dx,
        dy,
        x_arg,
        x_arg.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return dx
