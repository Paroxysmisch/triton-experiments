import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 16}, num_warps=2),
        triton.Config({"BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_N": 16}, num_warps=8),
        triton.Config({"BLOCK_N": 32}, num_warps=2),
        triton.Config({"BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_N": 64}, num_warps=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_N": 128}, num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_N": 256}, num_warps=2),
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=8),
    ],
    key=["n_rows", "n_cols"],
)
@triton.jit
def _swiglu_bwd_kernel(
    X,
    Y,
    DX,
    DY,
    DOUT,
    OUT,
    stride_x_row,
    stride_y_row,
    stride_dx_row,
    stride_dy_row,
    stride_dout_row,
    stride_out_row: tl.constexpr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    row_start = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_N
    col_offsets = col_start + tl.arange(0, BLOCK_N)
    mask = col_offsets < n_cols

    row_X = tl.load(X + row_start * stride_x_row + col_offsets, mask=mask, other=0)
    row_Y = tl.load(Y + row_start * stride_y_row + col_offsets, mask=mask, other=0)
    row_DOut = tl.load(DOUT + row_start * stride_dout_row + col_offsets, mask=mask, other=0)

    sigmoid_row_x = tl.sigmoid(row_X)
    silu_row_x = row_X * sigmoid_row_x
    row_DX = row_DOut * silu_row_x

    row_DY = row_DOut * (silu_row_x * (1 - sigmoid_row_x) + sigmoid_row_x) * row_Y

    tl.store(DX + row_start * stride_dx_row + col_offsets, row_DX, mask=mask)
    tl.store(DY + row_start * stride_dy_row + col_offsets, row_DY, mask=mask)

    if RECOMPUTE_OUTPUT:
        row_OUT = row_X * row_Y
        tl.store(OUT + row_start * stride_out_row + col_offsets, row_OUT, mask=mask)


def _swiglu_bwd(xy, dout, recompute_output=False):
    assert xy.is_contiguous()
    assert dout.is_contiguous()

    xy = xy.reshape([-1, xy.shape[-1]])
    dout = dout.reshape([-1, dout.shape[-1]])
    n_rows, n_cols = xy.shape
    x, y = xy.chunk(2, axis=-1)
    dx = torch.empty_like(x)
    dy = torch.empty_like(y)
    out = torch.empty_like(x) if recompute_output else None

    grid = lambda META: (
        triton.cdiv(n_rows, META["BLOCK_N"]),
        triton.cdiv(n_cols, META["BLOCK_N"]),
    )
    _swiglu_bwd_kernel[grid](
        x,
        y,
        dx,
        dy,
        dout,
        out,
        x.stride(0),
        y.stride(0),
        dx.stride(0),
        dy.stride(0),
        dout.stride(0),
        out.stride(0) if out is not None else 0,
        n_rows,
        n_cols,
        RECOMPUTE_OUTPUT=recompute_output,
    )
    return (dx, dy, out) if out is not None else (dx, dy)
