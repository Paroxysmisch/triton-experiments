import torch
import triton
import triton.language as tl


# ------------------------------------------------------------------------------------
#  _layer_norm_fwd_fused
#
#  Computes:
#     mean = sum(X[i]) / N
#     var  = sum((X[i] - mean)^2) / N
#     rstd = 1 / sqrt(var + eps)
#     Y[i] = ((X[i] - mean) * rstd) * W[i] + B[i]
#
#  Additionally stores mean and rstd for use in backward.
# ------------------------------------------------------------------------------------
@triton.jit
def _layer_norm_fwd_fused(
    X,            # ptr to input
    Y,            # ptr to output
    W,            # ptr to scale
    B,            # ptr to bias
    Mean,         # ptr to save mean
    Rstd,         # ptr to save rstd
    stride_m,     # leading dimension stride of X
    stride_n,     # second dimension stride of X
    M,            # number of rows
    N,            # number of features
    eps,          # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Each block processes one row
    # Return early if row is out of bounds
    if row_id >= M:
        return

    # Pointers to start of this row in X, Y
    x_row_ptr = X + row_id * stride_m
    y_row_ptr = Y + row_id * stride_m

    # 1) Compute mean
    # We'll accumulate partial sums in registers
    sum_ = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # Loop over blocks of columns
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_ += x
    # Sum partial sums into a single scalar
    mean = tl.sum(sum_) / N

    # 2) Compute var
    var_ = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        var_ += diff * diff
    var_val = tl.sum(var_) / N

    rstd_val = 1.0 / tl.sqrt(var_val + eps)
    # Store mean and rstd for backward
    tl.store(Mean + row_id, mean)
    tl.store(Rstd + row_id, rstd_val)

    # 3) Apply layernorm: Y[i] = (X[i] - mean) * rstd * W[i] + B[i]
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0)
        b = tl.load(B + cols, mask=mask, other=0.0)
        out = (x - mean) * rstd_val * w + b
        tl.store(y_row_ptr + cols, out, mask=mask)


# ------------------------------------------------------------------------------------
#  _layer_norm_bwd_dx_fused
#
#  Computes partial gradients for X, as well as partial sums for dW and dB.
#     dX[i] = (1.0 / N) * rstd * (N * dY[i] - sum(dY[i]) - (X[i] - mean) * rstd^2 * sum(dY[i]*(X[i]-mean)))
#             all multiplied by W[i]
#  Also accumulates partial dW and dB in intermediate buffers. Must be reduced later.
# ------------------------------------------------------------------------------------
@triton.jit
def _layer_norm_bwd_dx_fused(
    X,          # ptr to input
    Y_grad,     # ptr to gradient wrt output
    W,          # ptr to scale
    Mean,       # ptr to saved mean
    Rstd,       # ptr to saved rstd
    dX,         # ptr to gradient wrt input
    dW_buf,     # ptr to partial accum for grad_w
    dB_buf,     # ptr to partial accum for grad_b
    stride_m,   # leading dimension stride
    M,          # number of rows
    N,          # number of features
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    if row_id >= M:
        return

    # Load mean and rstd
    mean = tl.load(Mean + row_id)
    rstd_val = tl.load(Rstd + row_id)

    # Pointers for row
    x_row_ptr = X + row_id * stride_m
    dy_row_ptr = Y_grad + row_id * stride_m
    dx_row_ptr = dX + row_id * stride_m
    dw_buf_ptr = dW_buf + row_id * stride_m
    db_buf_ptr = dB_buf + row_id * stride_m

    # We'll need partial sums for the backward formula
    partial_sum_dy = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    partial_sum_xdy = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 1) Accumulate partial sums: sum(dy) and sum(dy*(x-mean))
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        partial_sum_dy += dy
        partial_sum_xdy += dy * (x - mean)
    sum_dy = tl.sum(partial_sum_dy)
    sum_xdy = tl.sum(partial_sum_xdy)

    # 2) Compute dX, partial dW, partial dB
    #    dW_partial = dY * (X-mean)*rstd
    #    dB_partial = dY
    #    dX = W[i] * rstd_val * (dY[i] - (1/N)*sum_dy - (x-mean)*rstd_val^2* (1/N)*sum_xdy )
    inv_n = 1.0 / N
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0)

        dB_val = dy
        dW_val = dy * (x - mean) * rstd_val
        # dX
        t1 = dy - sum_dy * inv_n
        t2 = (x - mean) * rstd_val * (sum_xdy * inv_n) * rstd_val
        dx_val = w * rstd_val * (t1 - t2)

        tl.store(dw_buf_ptr + cols, dW_val, mask=mask)
        tl.store(db_buf_ptr + cols, dB_val, mask=mask)
        tl.store(dx_row_ptr + cols, dx_val, mask=mask)


# ------------------------------------------------------------------------------------
#  _layer_norm_bwd_dwdb
#
#  Performs the final reduction over partial dW, dB across all rows, storing
#  the final results in dw_out, db_out.
# ------------------------------------------------------------------------------------
@triton.jit
def _layer_norm_bwd_dwdb(
    dW_buf,     # ptr to partial accum for grad_w
    dB_buf,     # ptr to partial accum for grad_b
    dW_out,     # ptr to final grad_w
    dB_out,     # ptr to final grad_b
    stride_m,   # leading dimension stride
    M,          # number of rows
    N,          # number of features
    BLOCK_SIZE: tl.constexpr
):
    # Each program_id(0) processes a range of columns
    col_off = tl.program_id(0) * BLOCK_SIZE
    cols = col_off + tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # We'll sum across M
    dW_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    dB_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for row_id in range(0, M):
        dW_row_ptr = dW_buf + row_id * stride_m
        dB_row_ptr = dB_buf + row_id * stride_m
        wval = tl.load(dW_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bval = tl.load(dB_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dW_sum += wval
        dB_sum += bval

    tl.store(dW_out + cols, dW_sum, mask=mask)
    tl.store(dB_out + cols, dB_sum, mask=mask)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, eps=1e-5, block_size=1024):
        """
        x: [M, N] input
        w: [N] scale
        b: [N] bias
        """
        M, N = x.shape
        # Create output tensors
        y = torch.empty_like(x)
        mean = torch.empty(M, dtype=x.dtype, device=x.device)
        rstd = torch.empty(M, dtype=x.dtype, device=x.device)

        # Launch forward kernel
        grid = (M,)
        tr
