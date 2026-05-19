import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X_ptr, Y_ptr, RSTD_ptr,
    stride_x: tl.int32,
    stride_y: tl.int32,
    N: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_x = X_ptr + row_idx * stride_x
    row_start_y = Y_ptr + row_idx * stride_y

    # -------------------------
    # 1. Compute sum of squares
    # -------------------------
    sq_sum = tl.float32(0.)
    for offset in range(0, N, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(row_start_x + idx, mask=mask, other=0.).to(tl.float32)
        sq_sum += tl.sum(x * x, where=mask)

    # Compute reciprocal sqrt
    rstd = 1.0 / tl.sqrt(sq_sum)

    # Store RSTD for this row
    tl.store(RSTD_ptr + row_idx, rstd)

    # -------------------------
    # 2. Normalize and store Y
    # -------------------------
    for offset in range(0, N, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(row_start_x + idx, mask=mask, other=0.).to(tl.float32)
        y = x * rstd
        tl.store(row_start_y + idx, y, mask=mask)

@triton.jit
def _l2_norm_bwd_kernel(
    X_ptr, DY_ptr, DX_ptr, RSTD_ptr,
    stride_x: tl.int32,
    stride_dy: tl.int32,
    stride_dx: tl.int32,
    N: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    rstd = tl.load(RSTD_ptr + row_idx)

    row_start_x = X_ptr + row_idx * stride_x
    row_start_dy = DY_ptr + row_idx * stride_dy
    row_start_dx = DX_ptr + row_idx * stride_dx

    # -----------------------
    # 1. Compute partial sums
    # -----------------------
    dot = tl.float32(0.)
    for offset in range(0, N, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(row_start_x + idx, mask=mask, other=0.).to(tl.float32)
        dy = tl.load(row_start_dy + idx, mask=mask, other=0.).to(tl.float32)
        dot += tl.sum(x * dy, where=mask)

    # --------------------------------
    # 2. Compute dX = rstd * dY - x * rstd^3 * (x dot dY)
    # --------------------------------
    rstd3 = rstd * rstd * rstd
    for offset in range(0, N, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(row_start_x + idx, mask=mask, other=0.).to(tl.float32)
        dy = tl.load(row_start_dy + idx, mask=mask, other=0.).to(tl.float32)
        dx = dy * rstd - x * rstd3 * dot
        tl.store(row_start_dx + idx, dx, mask=mask)

def _l2_norm_fwd(X: torch.Tensor):
    # Assumes X is 2D and contiguous in the last dimension
    assert X.dim() == 2, "Input X must be 2D"
    M, N = X.shape
    assert X.stride(-1) == 1, "Last dimension must be contiguous"
    # Prepare output
    Y = torch.empty_like(X)
    RSTD = torch.empty(M, dtype=X.dtype, device=X.device)
    BLOCK_SIZE = 1024

    grid = (M,)
    stride_x = X.stride(0)
    stride_y = Y.stride(0)

    _l2_norm_fwd_1pass_kernel[grid](
        X, Y, RSTD,
        stride_x, stride_y, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return Y, RSTD

def _l2_norm_bwd(X: torch.Tensor, DY: torch.Tensor, RSTD: torch.Tensor):
    assert X.dim() == 2 and DY.dim() == 2, "X and DY must be 2D"
    assert X.shape == DY.shape, "X and DY must match in shape"
    M, N = X.shape
    BLOCK_SIZE = 1024

    DX = torch.empty_like(X)
    grid = (M,)

    stride_x = X.stride(0)
    stride_dy = DY.stride(0)
    stride_dx = DX.stride(0)

    _l2_norm_bwd_kernel[grid](
        X, DY, DX, RSTD,
        stride_x, stride_dy, stride_dx, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return DX
