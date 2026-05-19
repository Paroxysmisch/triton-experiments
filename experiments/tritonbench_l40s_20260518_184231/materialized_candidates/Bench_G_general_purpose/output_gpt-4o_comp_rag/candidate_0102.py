import triton
import triton.language as tl
import paddle
from ..utils import calculate_settings

@triton.jit
def _swiglu_bwd_kernel(X_ptr, Y_ptr, DX_ptr, DY_ptr, DOUT_ptr, OUT_ptr, stride, n_cols: tl.constexpr, BLOCK_N: tl.constexpr, RECOMPUTE_OUTPUT: tl.constexpr):
    program_id = tl.program_id(0)
    X_ptr += program_id * stride
    Y_ptr += program_id * stride
    DX_ptr += program_id * stride
    DY_ptr += program_id * stride
    DOUT_ptr += program_id * stride
    if RECOMPUTE_OUTPUT:
        OUT_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < n_cols

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    Y_row = tl.load(Y_ptr + col_offsets, mask=mask, other=0)
    DOUT_row = tl.load(DOUT_ptr + col_offsets, mask=mask, other=0)
    
    if RECOMPUTE_OUTPUT:
        OUT_row = X_row * tl.sigmoid(X_row) * Y_row
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
    if recompute_output:
        OUT = paddle.empty_like(X)
    n_rows = X.shape[0]

    BLOCK_N, num_warps = calculate_settings(n_cols)

    _swiglu_bwd_kernel[(n_rows,)](
        X,
        Y,
        DX,
        DY,
        DOUT,
        OUT,
        X.strides[-2],
        n_cols=n_cols,
        BLOCK_N=BLOCK_N,
        RECOMPUTE_OUTPUT=recompute_output,
        num_warps=num_warps,
    )
    return DX.reshape(ori_shape), DY.reshape(ori_shape), OUT.reshape(ori_shape) if recompute_output else None
