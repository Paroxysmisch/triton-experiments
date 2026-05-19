import triton
import triton.language as tl
import paddle
from ..utils import calculate_settings

@triton.jit
def _swiglu_bwd_kernel(
    x_ptr, y_ptr, dx_ptr, dy_ptr, dout_ptr, out_ptr,
    stride_x, n_cols: tl.constexpr,
    BLOCK_N: tl.constexpr, RECOMPUTE_OUTPUT: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * stride_x

    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < n_cols

    # Load input data
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + row_start + col_offsets, mask=mask, other=0)
    dout = tl.load(dout_ptr + row_start + col_offsets, mask=mask, other=0)

    # Compute sigmoid and SiLU(X)
    sig_x = tl.sigmoid(x)
    silu_x = x * sig_x

    # Compute gradients for Y
    dy = dout * silu_x
    tl.store(dy_ptr + row_start + col_offsets, dy, mask=mask)

    # Compute gradient for X using the Swish derivative
    dsilu_dx = sig_x + x * sig_x * (1 - sig_x)
    dx = dout * y * dsilu_dx
    tl.store(dx_ptr + row_start + col_offsets, dx, mask=mask)

    # Optionally recompute and store the output
    if RECOMPUTE_OUTPUT:
        out = silu_x * y
        tl.store(out_ptr + row_start + col_offsets, out, mask=mask)

def _swiglu_bwd(dout, xy, recompute_output=False):
    # Ensure inputs are contiguous
    dout = dout.contiguous()
    xy = xy.contiguous()

    original_shape = xy.shape
    assert original_shape[-1] % 2 == 0, "Input tensor must have even last dimension"
    n_cols = original_shape[-1] // 2

    # Split the concatenated tensor into x and y
    x = xy[..., :n_cols]
    y = xy[..., n_cols:]

    # Reshape to 2D for kernel processing
    x_2d = x.reshape(-1, n_cols)
    y_2d = y.reshape(-1, n_cols)
    dout_2d = dout.reshape(-1, n_cols)

    # Allocate gradients
    dx = paddle.empty_like(x_2d)
    dy = paddle.empty_like(y_2d)

    # Allocate output tensor if needed
    out = paddle.empty_like(dout_2d) if recompute_output else None

    n_rows = x_2d.shape[0]

    # Determine block size and number of warps
    BLOCK_N, num_warps = calculate_settings(n_cols)

    # Kernel grid and launch
    grid = (n_rows,)
    _swiglu_bwd_kernel[grid](
        x_2d, y_2d, dx, dy, dout_2d,
        out if recompute_output else None,
        x_2d.stride(0), n_cols,
        BLOCK_N=BLOCK_N, RECOMPUTE_OUTPUT=recompute_output,
        num_warps=num_warps
    )

    # Reshape outputs back to original dimensions
    dx = dx.reshape(original_shape[:-1] + (n_cols,))
    dy = dy.reshape(original_shape[:-1] + (n_cols,))

    if recompute_output:
        out = out.reshape(dout.shape)
        return dx, dy, out
    else:
        return dx, dy
