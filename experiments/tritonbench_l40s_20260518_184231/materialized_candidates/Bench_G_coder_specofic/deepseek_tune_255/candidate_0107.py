import torch
import triton
import triton.language as tl

@triton.jit
def _swiglu_bwd_kernel(
    X,
    Y,
    DX,
    DY,
    DOUT,
    OUT,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    # Map program IDs to row indices
    row_start = tl.program_id(0) * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    # Load input slices
    x_mask = rows < M
    x = tl.load(X + rows * N + cols, mask=x_mask)
    y = tl.load(Y + rows * N + cols)
    dout = tl.load(DOUT + rows * N + cols)

    # Calculate derivatives
    dy = x * dout
    dx_sigmoid = y * dout
    dx = dx_sigmoid * (1.0 / (1.0 + tl.exp(-x)))

    # Store gradients
    tl.store(DX + rows * N + cols, dx, mask=x_mask)
    tl.store(DY + rows * N + cols, dy)

    # Optionally recompute and store output
    if RECOMPUTE_OUTPUT:
        out = x * tl.sigmoid(x)
        tl.store(OUT + rows * N + cols, out, mask=x_mask)


def _swiglu_bwd(xy, dout, dxy, recompute_output=False):
    # Ensure inputs are contiguous
    if not xy.is_contiguous():
        xy = xy.contiguous()
    if not dout.is_contiguous():
        dout = dout.contiguous()

    # Split xy into x and y
    x, y = xy.unbind(0)

    # Reshape for batch dimensions
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    dout = dout.reshape(-1, dout.shape[-1])
    M, N = x.shape

    # Allocate output tensors
    dx = torch.empty_like(x)
    dy = torch.empty_like(y)
    out = torch.empty_like(x) if recompute_output else None

    # Define grid
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    # Launch Triton kernel
    _swiglu_bwd_kernel[grid](
        x,
        y,
        dx,
        dy,
        dout,
        out,
        M,
        N,
        RECOMPUTE_OUTPUT=recompute_output,
    )

    # Squeeze out batch dimensions
    dx = dx.reshape(x.shape)
    dy = dy.reshape(y.shape)
    if recompute_output:
        out = out.reshape(x.shape)

    # Stack gradients and return
    dxy.data.set_((dx, dy))
    if recompute_output:
        return out
