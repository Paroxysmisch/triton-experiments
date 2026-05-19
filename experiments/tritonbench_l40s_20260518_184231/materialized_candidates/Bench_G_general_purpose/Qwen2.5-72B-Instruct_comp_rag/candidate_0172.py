import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X,  # pointer to the input
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    stride_x_row,  # stride for moving to the next row in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    DY += row * stride_x_row
    DX += row * stride_x_row

    _var = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x

    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)

        sum_dy_x = tl.sum(dy * x, axis=0)
        dx = dy * rstd - (sum_dy_x * (1 / (var + eps)) * rstd * x)
        tl.store(DX + cols, dx, mask=mask)

import torch

def _l2_norm_bwd(x, dy, eps):
    x = x.contiguous()
    dy = dy.contiguous()
    dx = torch.empty_like(x)

    x_arg = x.reshape(-1, x.shape[-1])
    dy_arg = dy.reshape(-1, dy.shape[-1])
    M, N = x_arg.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    num_warps = min(max(BLOCK_N // 256, 1), 8)
    _l2_norm_bwd_kernel[(M,)](
        x_arg,
        dy_arg,
        dx,
        x_arg.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    dx = dx.reshape_as(x)
    return dx
