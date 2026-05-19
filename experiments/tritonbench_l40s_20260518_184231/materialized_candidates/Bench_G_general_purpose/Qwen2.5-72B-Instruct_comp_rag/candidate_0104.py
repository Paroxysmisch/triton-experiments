import triton
import triton.language as tl
import paddle

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def _swiglu_bwd_kernel(
    X_ptr, Y_ptr, DX_ptr, DY_ptr, DOUT_ptr, OUT_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr, RECOMPUTE_OUTPUT: tl.constexpr
):
    program_id = tl.program_id(0)
    X_ptr += program_id * stride
    Y_ptr += program_id * stride
    DX_ptr += program_id * stride
    DY_ptr += program_id * stride
    DOUT_ptr += program_id * stride
    if RECOMPUTE_OUTPUT:
        OUT_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    Y_row = tl.load(Y_ptr + col_offsets, mask=mask, other=0)
    DOUT_row = tl.load(DOUT_ptr + col_offsets, mask=mask, other=0)

    if RECOMPUTE_OUTPUT:
        sig_X = tl.sigmoid(X_row)
        OUT_row = silu(X_row) * Y_row
        tl.store(OUT_ptr + col_offsets, OUT_row, mask=mask)
    else:
        OUT_row = tl.load(OUT_ptr + col_offsets, mask=mask, other=0)

    sig_X = tl.sigmoid(X_row)
    silu_X = X_row * sig_X
    dY_row = DOUT_row * silu_X
    dX_row = DOUT_row * (silu_X * (1 - sig_X) + sig_X) * Y_row

    tl.store(DX_ptr + col_offsets, dX_row, mask=mask)
    tl.store(DY_ptr + col_offsets, dY_row, mask=mask)

def _swiglu_bwd(X, Y, DX, DY, DOUT, OUT=None, recompute_output=False):
    ori_shape = X.shape
    n_cols = ori_shape[-1]
    X = X.reshape([-1, n_cols])
    Y = Y.reshape([-1, n_cols])
    DX = DX.reshape([-1, n_cols])
    DY = DY.reshape([-1, n_cols])
    DOUT = DOUT.reshape([-1, n_cols])
    if OUT is not None:
        OUT = OUT.reshape([-1, n_cols])

    n_rows = X.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    if recompute_output:
        OUT = paddle.empty_like(X)

    _swiglu_bwd_kernel[(n_rows,)](
        X,
        Y,
        DX,
        DY,
        DOUT,
        OUT,
        X.strides[-2],
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        RECOMPUTE_OUTPUT=recompute_output
    )

    DX = DX.reshape(ori_shape)
    DY = DY.reshape(ori_shape)
    if recompute_output:
        OUT = OUT.reshape(ori_shape)
        return DX, DY, OUT
    else:
        return DX, DY
