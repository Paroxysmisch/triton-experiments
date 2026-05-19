import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_forward_kernel(
    X,  # input tensor
    Y,  # output tensor
    W,  # weight tensor
    B,  # bias tensor
    Mean,  # mean tensor
    Rstd,  # reciprocal of std tensor
    stride_x,  # stride for input
    N,  # number of columns
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # block size
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x
    X_row = X + row_start
    Y_row = Y + row_start

    # Compute mean
    mean = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    # Compute variance
    var = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        x_centered = x - mean
        var += tl.sum(x_centered * x_centered, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Save mean and rstd
    tl.store(Mean + row_idx, mean)
    tl.store(Rstd + row_idx, rstd)

    # Normalize and apply weights and biases
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y_row + cols, y, mask=mask)

@triton.jit
def _layer_norm_backward_kernel(
    DX, DY, X, W, Mean, Rstd, DW_partial, DB_partial, Lock,
    stride_x, N, GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x
    X_row = X + row_start
    DY_row = DY + row_start
    DX_row = DX + row_start

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load data for this block
    x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean + row_idx)
    rstd = tl.load(Rstd + row_idx)

    x_hat = (x - mean) * rstd
    wdy = w * dy

    # Compute dx
    c1 = tl.sum(x_hat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (x_hat * c1 + c2)) * rstd
    tl.store(DX_row + cols, dx, mask=mask)

    # Compute partial derivatives for weights and biases
    partial_dw = (dy * x_hat).to(w.dtype)
    partial_db = dy.to(w.dtype)

    # Accumulate partial sums using locks
    lock_id = row_idx % GROUP_SIZE_M
    Lock_ptr = Lock + lock_id
    DW_ptr = DW_partial + lock_id * N + cols
    DB_ptr = DB_partial + lock_id * N + cols

    # Atomic update to prevent data races
    while tl.atomic_cas(Lock_ptr, 0, 1) != 0:
        pass
    current_dw = tl.load(DW_ptr, mask=mask, other=0.0)
    current_db = tl.load(DB_ptr, mask=mask, other=0.0)
    tl.store(DW_ptr, current_dw + partial_dw, mask=mask)
    tl.store(DB_ptr, current_db + partial_db, mask=mask)
    tl.atomic_xchg(Lock_ptr, 0)

@triton.jit
def _sum_partials_kernel(
    DW_partial, DB_partial, DW, DB, M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    col_idx = tl.program_id(0) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = col_idx < N

    sum_dw = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    sum_db = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for row_block in range(0, M, BLOCK_SIZE_M):
        rows = row_block + tl.arange(0, BLOCK_SIZE_M)
        mask_rows = rows < M
        mask = mask_rows[:, None] & mask_cols[None, :]

        offs = rows[:, None] * N + col_idx[None, :]
        dw = tl.load(DW_partial + offs, mask=mask, other=0.0)
        db = tl.load(DB_partial + offs, mask=mask, other=0.0)

        sum_dw += tl.sum(dw, axis=0)
        sum_db += tl.sum(db, axis=0)

    tl.store(DW + col_idx, sum_dw, mask=mask_cols)
    tl.store(DB + col_idx, sum_db, mask=mask_cols)

def layer_norm_forward(x, weight, bias, eps):
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    num_warps = 4 if BLOCK_SIZE <= 2048 else 8

    grid = (M,)
    _layer_norm_forward_kernel[grid](
        x_2d, y, weight, bias, mean, rstd,
        x_2d.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )
    return y, mean, rstd

def layer_norm_backward(dy, x, weight, mean, rstd):
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape

    GROUP_SIZE_M = 64
    if N <= 8192:
        GROUP_SIZE_M = 96
    elif N <= 4096:
        GROUP_SIZE_M = 128
    elif N <= 1024:
        GROUP_SIZE_M = 256

    locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=x.device)
    dw_partial = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=x.device)
    db_partial = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=x.device)
    dx = torch.empty_like(x)

    BLOCK_SIZE = 128
    num_warps = 4 if BLOCK_SIZE <= 128 else 8

    grid = (M,)
    _layer_norm_backward_kernel[grid](
        dx, dy, x_2d, weight, mean, rstd,
        dw_partial, db_partial, locks,
        x_2d.stride(0), N,
        GROUP_SIZE_M=GROUP_SIZE_M,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    dw = torch.empty_like(weight)
    db = torch.empty_like(bias)
    grid_sum = (triton.cdiv(N, 128),)
    _sum_partials_kernel[grid_sum](
        dw_partial, db_partial, dw, db,
        min(GROUP_SIZE_M, M), N,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=128
    )

    return dx, dw, db

class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        y, mean, rstd = layer_norm_forward(x, weight, bias, eps)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, _, mean, rstd = ctx.saved_tensors
        dx, dw, db = layer_norm_backward(dy, x, weight, mean, rstd)
        return dx, dw, db, None

# Example usage
def layer_norm(x, normalized_shape, weight, bias, eps=1e-5):
    return LigerLayerNormFunction.apply(x, weight, bias, eps)
