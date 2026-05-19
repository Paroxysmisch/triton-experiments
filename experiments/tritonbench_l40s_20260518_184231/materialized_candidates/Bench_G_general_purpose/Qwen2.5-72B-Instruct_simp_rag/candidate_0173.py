import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X,  # pointer to the input
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    DY += row * stride
    DX += row * stride

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x

    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        c1 = tl.sum(dy * x_hat, axis=0) / N
        c2 = tl.sum(dy, axis=0) / N
        dx = (dy - (x_hat * c1 + c2)) * rstd
        tl.store(DX + cols, dx, mask=mask)

import torch

def _l2_norm_bwd(x, dy, eps=1e-6):
    x = x.contiguous()
    dy = dy.contiguous()
    dx = torch.empty_like(x)

    x_arg = x.reshape(-1, x.shape[-1])
    dy_arg = dy.reshape(-1, dy.shape[-1])
    dx_arg = dx.reshape(-1, dx.shape[-1])

    M, N = x_arg.shape
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _l2_norm_bwd_kernel[(M,)](
        x_arg,
        dy_arg,
        dx_arg,
        x_arg.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return dx
