import triton
import triton.language as tl
import paddle
from ..utils import calculate_settings

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def _swiglu_bwd_kernel(
    X_ptr, Y_ptr, DX_ptr, DY_ptr, DOUT_ptr, OUT_ptr,
    stride, n_cols: tl.constexpr, BLOCK_N: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr
):
    pid = tl.program_id(0)
    X_ptr += pid * stride
    Y_ptr += pid * stride
    DX_ptr += pid * stride
    DY_ptr += pid * stride
    DOUT_ptr += pid * stride

    # Only adjust OUT pointer if recomputing output
    if RECOMPUTE_OUTPUT:
        OUT_ptr += pid * stride

    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < n_cols

    # Load input slices
    x_val = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y_val = tl.load(Y_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    dout_val = tl.load(DOUT_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Optionally recompute output
    if RECOMPUTE_OUTPUT:
        out_val = silu(x_val) * y_val
        tl.store(OUT_ptr + col_offsets, out_val, mask=mask)
    else:
        out_val = tl.load(OUT_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute gradients with respect to X and Y
    sig_x = tl.sigmoid(x_val)
    silu_x = x_val * sig_x
    dsilu_dx = sig_x + x_val * sig_x * (1 - sig_x)

    dx_val = dout_val * dsilu_dx * y_val
    dy_val = dout_val * silu_x

    # Write back gradients
    tl.store(DX_ptr + col_offsets, dx_val, mask=mask)
    tl.store(DY_ptr + col_offsets, dy_val, mask=mask)

def _swiglu_bwd(x, y, dout, recompute_output=False):
    # Ensure inputs are contiguous
    x = x.contiguous()
    y = y.contiguous()
    dout = dout.contiguous()

    # Reshape input tensors for 2D processing (batch dimension)
    original_shape = x.shape
    rows = 1
    cols = original_shape[-1]

    # If more than one dimension, flatten all except the last
    if len(original_shape) > 1:
        rows = 1
        for dim in original_shape[:-1]:
            rows *= dim

    x_2d = x.reshape([rows, cols])
    y_2d = y.reshape([rows, cols])
    dout_2d = dout.reshape([rows, cols])

    # Allocate space for gradients
    dx = paddle.empty_like(x_2d)
    dy = paddle.empty_like(y_2d)
    # Allocate optional recomputed output
    out = None
    if recompute_output:
        out = paddle.empty_like(x_2d)

    # Calculate kernel settings
    BLOCK_N, num_warps = calculate_settings(cols)

    # Launch the Triton kernel
    _swiglu_bwd_kernel[(rows,)](
        x_2d,
        y_2d,
        dx,
        dy,
        dout_2d,
        out if recompute_output else dout_2d,  # pass something valid
        x_2d.strides[-2],
        n_cols=cols,
        BLOCK_N=BLOCK_N,
        RECOMPUTE_OUTPUT=recompute_output,
        num_warps=num_warps
    )

    # Reshape to original shape
    dx = dx.reshape(original_shape)
    dy = dy.reshape(original_shape)
    return dx, dy, out.reshape(original_shape) if recompute_output else None
