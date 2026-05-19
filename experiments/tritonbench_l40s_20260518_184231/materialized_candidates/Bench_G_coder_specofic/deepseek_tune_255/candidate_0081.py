import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_fused(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    D_GROUP_SIZE_M: tl.constexpr,
):
    """
    LayerNorm forward kernel
    Arguments:
        X: input tensor with shape (N, ), data type is tl.float32.
        Y: output tensor with shape (N, ), data type is tl.float32.
        W: weight tensor with shape (N, ), data type is tl.float32.
        B: bias tensor with shape (N, ), data type is tl.float32.
        Mean: mean tensor with shape (D_GROUP_SIZE_M, ), data type is tl.float32.
        Rstd: 1/std tensor with shape (D_GROUP_SIZE_M, ), data type is tl.float32.
        stride: stride of X.
        N: the normalization dimension.
        eps: epsilon to avoid division by zero.
        BLOCK_SIZE: block size for triton kernel.
        GROUP_SIZE_M: group size for triton kernel.
        D_GROUP_SIZE_M: the ceil of the normalization dimension divided by GROUP_SIZE_M.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_zm = tl.where(mask, x - mean, 0.0)
    tl.store(Mean + row, mean)

    x_var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(x_var + eps)
    tl.store(Rstd + row, rstd)

    x_hat = x_zm * rstd
    y = x_hat * w + b

    tl.store(Y + row * stride + cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused(
    DX,
    DY,
    X,
    W,
    Mean,
    Rstd,
    DW,
    DB,
    Lock,
    stride,
    N,
    GROUP_SIZE_M: tl.constexpr,
    D_GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm backward kernel for dx
    Arguments:
        DX: input gradient tensor with shape (N, ), data type is tl.float32.
        DY: output gradient tensor with shape (N, ), data type is tl.float32.
        X: input tensor with shape (N, ), data type is tl.float32.
        W: weight tensor with shape (N, ), data type is tl.float32.
        Mean: mean tensor with shape (D_GROUP_SIZE_M, ), data type is tl.float32.
        Rstd: 1/std tensor with shape (D_GROUP_SIZE_M, ), data type is tl.float32.
        DW: weight gradient tensor with shape (N, ), data type is tl.float32.
        DB: bias gradient tensor with shape (N, ), data type is tl.float32.
        Lock: lock for synchronization.
        stride: stride of X.
        N: the normalization dimension.
        GROUP_SIZE_M: group size for triton kernel.
        D_GROUP_SIZE_M: the ceil of the normalization dimension divided by GROUP_SIZE_M.
        BLOCK_SIZE: block size for triton kernel.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    x_hat = (x - mean) * rstd
    wdy = w * dy
    x_hat = tl.where(mask, x_hat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)

    mean1 = tl.sum(x_hat * wdy, axis=0) / N
    mean2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (x_hat * mean1 + mean2)) * rstd

    tl.store(DX + row * stride + cols, dx, mask=mask)

    # Parameters' gradients
    lock_id = tl.program_id(0) % D_GROUP_SIZE_M
    dy_group = tl.load(DY + lock_id * BLOCK_SIZE * stride + cols, mask=mask, other=0.0).to(tl.float32)
    x_group = tl.load(X + lock_id * BLOCK_SIZE * stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean_group = tl.load(Mean + lock_id)
    rstd_group = tl.load(Rstd + lock_id)

    x_hat = (x_group - mean_group) * rstd_group
    wdy = w * dy_group

    mean1 = tl.sum(x_hat * wdy, axis=0) / N
    mean2 = tl.sum(wdy, axis=0) / N

    # Lock to guarantee that only one workgroup write the gradients at the same time.
    dw = tl.sum(dy_group * x_hat, axis=0)
    db = tl.sum(dy_group, axis=0)

    num_workgroups = D_GROUP_SIZE_M
    for i in range(0, num_workgroups):
        lock_id = tl.load(Lock + lock_id)
        tl.atomic_add(DW + lock_id * BLOCK_SIZE + cols, dw, mask=mask)
        tl.atomic_add(DB + lock_id * BLOCK_SIZE + cols, db, mask=mask)
        lock_id = (lock_id + 1) % num_workgroups
        tl.store(Lock + row, lock_id)


@triton.jit
def _layer_norm_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    LayerNorm backward kernel for dw and db
    Arguments:
        DW: weight gradient tensor with shape (M, N), data type is tl.float32.
        DB: bias gradient tensor with shape (M, N), data type is tl.float32.
        FINAL_DW: final weight gradient tensor with shape (M, N), data
