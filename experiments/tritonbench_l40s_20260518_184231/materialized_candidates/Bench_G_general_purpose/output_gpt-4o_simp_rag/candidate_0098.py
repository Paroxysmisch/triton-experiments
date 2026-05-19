import triton
import triton.language as tl
import paddle
from ..utils import calculate_settings

@triton.jit
def _swiglu_bwd_kernel(X_ptr, Y_ptr, DOUT_ptr, DX_ptr, DY_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0)
    X_ptr += program_id * stride
    Y_ptr += program_id * stride
    DOUT_ptr += program_id * stride
    DX_ptr += program_id * stride
    DY_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    Y_row = tl.load(Y_ptr + col_offsets, mask=mask, other=0)
    DOUT_row = tl.load(DOUT_ptr + col_offsets, mask=mask, other=0)

    sig_X = tl.sigmoid(X_row)
    silu_X = X_row * sig_X

    DY_row = DOUT_row * silu_X
    DX_row = DOUT_row * (silu_X * (1 - sig_X) + sig_X) * Y_row

    tl.store(DX_ptr + col_offsets, DX_row, mask=mask)
    tl.store(DY_ptr + col_offsets, DY_row, mask=mask)

def _swiglu_bwd(X, Y, DOUT, OUT=None):
    ori_shape = X.shape
    n_cols = ori_shape[-1]
    X = X.reshape([-1, n_cols])
    Y = Y.reshape([-1, n_cols])
    DOUT = DOUT.reshape([-1, n_cols])
    DX = paddle.empty_like(X)
    DY = paddle.empty_like(Y)
    n_rows = X.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_bwd_kernel[(n_rows,)](
        X,
        Y,
        DOUT,
        DX,
        DY,
        X.strides[-2],
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return DX.reshape(ori_shape), DY.reshape(ori_shape)
