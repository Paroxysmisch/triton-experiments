import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X,
    DY,
    DX,
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    # map program id to the row of X and DY it should compute
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * BLOCK_N
    DY += row * BLOCK_N
    # compute variance
    _var = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # compute dx
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (dy) * rstd
        tl.store(DX + X + cols, x_hat, mask=mask)

def _l2_norm_bwd(x, dy, eps):
    # reshape x and dy to 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    BLOCK_SIZE = min(
        max(x.stride(0), x.stride(1), dy.stride(0), dy.stride(1)), triton.next_power_of_2(N)
    )
    if BLOCK_SIZE * 2 > x.stride(0) or BLOCK_SIZE * 2 > x.stride(1):
        x = x.contiguous()
        dy = dy.contiguous()
    dx = torch.empty_like(dy)
    assert (
        triton.cdiv(N, BLOCK_SIZE) * BLOCK_SIZE == N
    ), "This layer norm doesn't support feature dim != 0 mod block_size."
    grid = (M, triton.cdiv(N, BLOCK_SIZE))
    sigma = BLOCK_SIZE / (12.0**0.5)
    _l2_norm_bwd_kernel[grid](
        x_arg,
        dy,
        dx,
        x_arg.stride(0),
        N,
        eps,
        BLOCK_SIZE,
    )
    return dx
