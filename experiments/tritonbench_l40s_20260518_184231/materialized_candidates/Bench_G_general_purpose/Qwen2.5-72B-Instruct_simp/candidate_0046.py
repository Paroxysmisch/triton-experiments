import triton
import triton.language as tl

# Triton kernel for L2 normalization forward pass
@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X, Y, M, N, stride_xm, stride_xn, stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_M
    offsets_m = block_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = X + (offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn)
    y_ptrs = Y + (offsets_m[:, None] * stride_ym + offsets_n[None, :] * stride_yn)

    # Load data
    x = tl.load(x_ptrs, mask=offsets_m[:, None] < M, other=0.0)

    # Compute L2 norm
    l2_norm = tl.sqrt(tl.sum(x * x, axis=1))

    # Normalize
    y = x / l2_norm[:, None]

    # Store result
    tl.store(y_ptrs, y, mask=offsets_m[:, None] < M)

# Triton kernel for L2 normalization backward pass
@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX, M, N, stride_xm, stride_xn, stride_dym, stride_dyn, stride_dxm, stride_dxn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_M
    offsets_m = block_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = X + (offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn)
    dy_ptrs = DY + (offsets_m[:, None] * stride_dym + offsets_n[None, :] * stride_dyn)
    dx_ptrs = DX + (offsets_m[:, None] * stride_dxm + offsets_n[None, :] * stride_dxn)

    # Load data
    x = tl.load(x_ptrs, mask=offsets_m[:, None] < M, other=0.0)
    dy = tl.load(dy_ptrs, mask=offsets_m[:, None] < M, other=0.0)

    # Compute L2 norm
    l2_norm = tl.sqrt(tl.sum(x * x, axis=1))

    # Compute gradient w.r.t. input
    dx = (dy - (tl.sum(dy * x, axis=1) / (l2_norm * l2_norm))[:, None] * x) / l2_norm[:, None]

    # Store result
    tl.store(dx_ptrs, dx, mask=offsets_m[:, None] < M)

import torch

def _l2_norm_fwd(X):
    M, N = X.shape
    Y = torch.empty_like(X)
    grid = (triton.cdiv(M, 128),)
    _l2_norm_fwd_1pass_kernel[grid](
        X, Y, M, N, X.stride(0), X.stride(1), Y.stride(0), Y.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=N
    )
    return Y

def _l2_norm_bwd(X, DY):
    M, N = X.shape
    DX = torch.empty_like(X)
    grid = (triton.cdiv(M, 128),)
    _l2_norm_bwd_kernel[grid](
        X, DY, DX, M, N, X.stride(0), X.stride(1), DY.stride(0), DY.stride(1), DX.stride(0), DX.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=N
    )
    return DX
