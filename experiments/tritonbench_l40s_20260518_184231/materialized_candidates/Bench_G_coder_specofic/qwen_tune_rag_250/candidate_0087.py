output, mask = mask)

@triton.jit
def _rms_layernorm_backward(
    dY, dY_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    dW, dW_row_stride,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr
):
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0)#.to(tl.float32)
    inv_var = tl.load(r).to(tl.float32)

    normed = X_row * inv_var
    w_dot_dy = tl.sum(normed * dY_row, axis = 0)
    dw = dY_row * inv_var
    dw *= W_row
    dw *= normed
    dw -= inv_var * w_dot_dy * normed
    tl.store(dW + col_offsets, dw, mask = mask)

def rms_layernorm_forward_triton(x, weight, eps):
    out = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    out_arg = out.reshape(-1, out.shape[-1])
    weight_arg = weight.reshape(-1, weight.shape[-1])

    n_rows, n_cols = x_arg.shape
    r = torch.empty(n_rows, dtype = torch.float32, device = "cuda")

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _rms_layernorm_forward[(n_rows,)](
        out_arg, out_arg.stride(0),
        x_arg, x_arg.stride(0),
        weight_arg, weight_arg.stride(0),
        r, r.stride(0),
        n_cols, eps,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps  = num_warps,
    )
    return out, out.squeeze(), r

def rms_layernorm_backward_triton(dY, X, W, r, r_mean):
    dX = torch.empty_like(X)
    dX = dX.reshape(-1, dX.shape[-1])
    n_rows, n_cols = dX.shape

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _rms_layernorm_backward[(n_rows,)](
        dY, dY.stride(0),
        X, X.stride(0),
        W, W.stride(0),
        r, r.stride(0),
        dX, dX.stride(0),
        n_cols, eps,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps  = num_warps,
    )
    if len(dY.shape) > 2:
        dX = dX.reshape(X.shape)
    return dX
