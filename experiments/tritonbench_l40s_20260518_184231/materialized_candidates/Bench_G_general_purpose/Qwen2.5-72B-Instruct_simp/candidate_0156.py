import triton
import triton.language as tl

# Triton kernel for the softmax operation
@triton.jit
def _softmax(Y, X, M, L, S, stride_yb, stride_yh, stride_yl, stride_xb, stride_xh, stride_xl, stride_mb, stride_mh, stride_ml, n_ctx, mask_type, causal, log, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    bid = pid // (n_ctx // BLOCK_SIZE)
    hid = pid % (n_ctx // BLOCK_SIZE)
    row = bid * n_ctx + hid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, BLOCK_SIZE)
    mask = row < n_ctx
    X_row = X + (bid * stride_yb + hid * stride_yh + col * stride_yl)
    Y_row = Y + (bid * stride_yb + hid * stride_yh + col * stride_yl)
    M_row = M + (bid * stride_mb + hid * stride_mh + col * stride_ml) if M is not None else None

    x = tl.load(X_row, mask=mask, other=-float('inf'))
    if M_row is not None:
        m = tl.load(M_row, mask=mask, other=0.0)
        x = x * m
    if causal:
        causal_mask = col >= row
        x = tl.where(causal_mask, x, -float('inf'))

    max_val = tl.max(x, axis=0)
    x = x - max_val
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    if log:
        y = x - tl.log(sum_exp)
    else:
        y = exp_x / sum_exp

    tl.store(Y_row, y, mask=mask)

# Triton kernel for the backward pass of the softmax operation
@triton.jit
def _softmax_backward(dY, Y, dX, stride_dyb, stride_dyh, stride_dyl, stride_yb, stride_yh, stride_yl, stride_dxb, stride_dyh, stride_dxl, n_ctx, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    bid = pid // (n_ctx // BLOCK_SIZE)
    hid = pid % (n_ctx // BLOCK_SIZE)
    row = bid * n_ctx + hid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, BLOCK_SIZE)
    mask = row < n_ctx
    dY_row = dY + (bid * stride_dyb + hid * stride_dyh + col * stride_dyl)
    Y_row = Y + (bid * stride_yb + hid * stride_yh + col * stride_yl)
    dX_row = dX + (bid * stride_dxb + hid * stride_dyh + col * stride_dxl)

    dY_val = tl.load(dY_row, mask=mask, other=0.0)
    Y_val = tl.load(Y_row, mask=mask, other=0.0)
    sum_dY = tl.sum(dY_val * Y_val, axis=0)
    dX_val = dY_val * Y_val - Y_val * sum_dY

    tl.store(dX_row, dX_val, mask=mask)

# Wrapper function for the softmax operation
def softmax(Y, X, M=None, L=None, S=None, mask_type=None, causal=False, log=False, BLOCK_SIZE=128):
    n_ctx = X.shape[-1]
    grid = (X.shape[0] * (n_ctx // BLOCK_SIZE),)
    _softmax[grid](Y, X, M, L, S, Y.stride(0), Y.stride(1), Y.stride(2), X.stride(0), X.stride(1), X.stride(2), M.stride(0) if M is not None else 0, M.stride(1) if M is not None else 0, M.stride(2) if M is not None else 0, n_ctx, mask_type, causal, log, BLOCK_SIZE)

# Wrapper function for the backward pass of the softmax operation
def softmax_backward(dY, Y, dX, BLOCK_SIZE=128):
    n_ctx = dY.shape[-1]
    grid = (dY.shape[0] * (n_ctx // BLOCK_SIZE),)
    _softmax_backward[grid](dY, Y, dX, dY.stride(0), dY.stride(1), dY.stride(2), Y.stride(0), Y.stride(1), Y.stride(2), dX.stride(0), dX.stride(1), dX.stride(2), n_ctx, BLOCK_SIZE)
