import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X,
    Y,
    Rstd,
    stride_m,
    stride_n,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # using triton 3.0.0rc1
    # modified from row-norm FOR COVERAGE
    row = tl.program_id(0)
    Y += row * stride_m
    X += row * stride_m

    _row_norm_square_i = tl.zeros([BLOCK_SIZE], tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x_i = tl.load(X + cols, mask=cols < N, other=0).to(tl.float32)
        _row_norm_square_i += x_i * x_i
    row_norm_square_i = tl.sum(_row_norm_square_i, axis=0)
    row_norm_i = tl.sqrt(row_norm_square_i + eps)

    tl.store(Rstd + row, row_norm_i)

    row_norm_i_recip = tl.reciprocal(row_norm_i)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_i = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        y_i = x_i * row_norm_i_recip
        tl.store(Y + cols, y_i, mask=mask)

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE": 128}, num_warps=4, num_stages=4, pre_hook=init_to_zero("Y")
        ),
    ],
    key=["M", "N"],
)
@triton.jit
def _l2_norm_bwd_kernel(
    X,
    DY,
    DX,
    Rstd,
    stride_xm,
    stride_dxm,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * stride_xm
    col_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < M * N  # added: to make num_stages = 1

    x_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    dy_row = tl.load(DY + col_offsets, mask=mask, other=0).to(tl.float32)
    norm = tl.load(Rstd + row)

    dx_row = (
        dy_row * norm
        + x_row * (tl.sum(x_row * dy_row, 0) / (norm * norm) - dy_row / norm)
    )

    tl.store(DX + col_offsets, dx_row, mask=mask)
