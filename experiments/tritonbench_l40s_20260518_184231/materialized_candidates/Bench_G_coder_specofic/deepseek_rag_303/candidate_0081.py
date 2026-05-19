import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "num_warps": 4}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "num_warps": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "num_warps": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "num_warps": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "num_warps": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "num_warps": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "num_warps": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "num_warps": 4}, num_stages=4, num_warps=4
        ),
    ],
    key=["M", "N"],
)
@triton.jit
def _layer_norm_fwd_fused(
    X,  # shape: [M, N]
    W,  # shape: [N]
    B,  # shape: [N]
    Y,  # shape: [M, N]
    Mean,  # shape: [M]
    Rstd,  # shape: [M]
    eps,
    stride_xm,  # how much to increase the pointer when moving by 1 row
    stride_xn,  # how much to increase the pointer when moving by 1 col
    stride_ym,
    stride_yn,
    M,
    N,  # must be divisible by 8
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Computes variance- and shift-normalized form of `x`:
    y[i, :] = (x[i, :] - E(x[i, :]; y[0:i, :]) / sqrt(var(x[i, :]; y[0:i, :]) + eps)
    x: (M, N)
    y: (M, N)
    w: (N)
    b: (N)
    """

    row = tl.program_id(0)
    col_block_id = tl.program_id(1)

    col_offsets = col_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    X += (row * stride_xm) * stride_xn + col_offsets
    Y += (row * stride_ym) * stride_yn + col_offsets
    W += col_offsets
    B += col_offsets

    row_group = row % GROUP_SIZE_M
    tl.static_print(row, col_block_id, BLOCK_SIZE_M, BLOCK_SIZE_N, row_group)

    mean = 0
    _mean = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE_N * 2):
        cols = off + col_offsets
        mask = cols < N

        x = tl.load(X, mask=mask, other=0.0).to(tl.float32)

        _mean += x

        X += BLOCK_SIZE_N * stride_xn

    mean = tl.sum(_mean, axis=0) / N
    tl.store(Mean + row, mean)

    var = 0
    _var = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE_N * 2):
        cols = off + col_offsets
        mask = cols < N

        x = tl.load(X - BLOCK_SIZE_N * stride_xn, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)

        _var += x * x

        X += BLOCK_SIZE_N * stride_xn

    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    for off in range(0, N, BLOCK_SIZE_N * 2):
        cols = off + col_offsets
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        b = tl.load(B + cols, mask=mask).to(tl.float32)
        # we have to use the reduced mean and rstd to compute y
        y = (
            tl.load(X - BLOCK_SIZE_N * stride_xn, mask=mask, other=0.0).to(tl.float32)
            - mean
        )
        y *= rstd
        y *= w
        y += b

        # write back
        mask = cols < N
        tl.store(Y - BLOCK_SIZE_N * stride_yn, y, mask=mask)

        X += BLOCK_SIZE_N * stride_xn
        Y += BLOCK_SIZE_N * stride_yn


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "num_warps": 4}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "num_warps": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "num_warps": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "num_warps": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "num_warps": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "num_warps": 4}, num_stages=4, num_warps=4
        ),
        trit
