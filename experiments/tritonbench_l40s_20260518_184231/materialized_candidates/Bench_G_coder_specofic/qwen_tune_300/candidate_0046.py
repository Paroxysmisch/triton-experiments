import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X, Y, stride_x_row, stride_y_row, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel performs L2 normalization on a per-row basis.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row

    # Load the row into SRAM, doing a blocking load to minimize bank conflicts.
    x = tl.load(X + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0).to(tl.float32)
    x = tl.where(tl.arange(0, BLOCK_SIZE) < N, x, 0.0)

    # Compute the variance
    x_mean = tl.sum(x, axis=0) / N
    x_zm = x - x_mean
    x_zm = tl.where(tl.arange(0, BLOCK_SIZE) < N, x_zm, 0.0)
    var = tl.sum(x_zm * x_zm, axis=0) / N
    var += eps
    rstd = 1 / tl.sqrt(var)

    # Normalize and apply the linear transformation
    y = x_zm * rstd
    y = tl.where(tl.arange(0, BLOCK_SIZE) < N, y, 0.0)
    tl.store(Y + tl.arange(0, BLOCK_SIZE), y, mask=tl.arange(0, BLOCK_SIZE) < N)

def _l2_norm_fwd(X, eps):
    # The wrapper for the forward pass kernel.
    if X.stride(-1) != 1:
        X = X.contiguous()

    feat_dim = X.shape[-1]
    assert feat_dim < 65536, "Feature dimension should be smaller than 64KB for Triton kernel memory constraints."

    Y = torch.empty_like(X)
    batch_dim = X.numel() // feat_dim

    with torch.cuda.device(X.device):
        _l2_norm_fwd_1pass_kernel[(batch_dim,)](
            X, Y,
            X.stride(0), Y.stride(0),
            feat_dim, eps,
            BLOCK_SIZE=triton.next_power_of_2(feat_dim),
        )
    return Y

@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX, stride_x_row, stride_dy_row, stride_dx_row, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel performs the backward pass for L2 normalization.
    row = tl.program_id(0)
    X += row * stride_x_row
    DY += row * stride_dy_row
    DX += row * stride_dx_row

    # Load the inputs and gradients for the row
    x = tl.load(X + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0).to(tl.float32)
    dy = tl.load(DY + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0).to(tl.float32)
    x = tl.where(tl.arange(0, BLOCK_SIZE) < N, x, 0.0)
    dy = tl.where(tl.arange(0, BLOCK_SIZE) < N, dy, 0.0)

    # Compute the gradients
    x_mean = tl.sum(x, axis=0) / N
    x_zm = x - x_mean
    var = tl.sum(x_zm * x_zm, axis=0) / N
    var += eps
    rstd = 1 / tl.sqrt(var)
    x_hat = x_zm * rstd
    sigma = rstd * (N - 1) / tl.sqrt(N * N - N)
    gamma = tl.load(X + N).to(tl.float32)
    dy_hat = dy * gamma
    dx_hat = dy_hat * x_hat - sigma * tl.sum(dy_hat * x_hat, axis=0)
    dx = dx_hat * rstd

    # Write out the gradients
    tl.store(DX + tl.arange(0, BLOCK_SIZE), dx, mask=tl.arange(0, BLOCK_SIZE) < N)

def _l2_norm_bwd(DY, X):
    # The wrapper for the backward pass kernel.
    if DY.stride(-1) != 1:
        DY = DY.contiguous()

    feat_dim = DY.shape[-1]
    assert feat_dim < 65536, "Feature dimension should be smaller than 64KB for Triton kernel memory constraints."

    DX = torch.empty_like(DY)
    batch_dim = DY.numel() // feat_dim

    with torch.cuda.device(DY.device):
        _l2_norm_bwd_kernel[(batch_dim,)](
            X, DY, DX,
            X.stride(0), DY.stride(0), DX.stride(0),
            feat_dim, 1e-5,
            BLOCK_SIZE=triton.next_power_of_2(feat_dim),
        )
    return DX
