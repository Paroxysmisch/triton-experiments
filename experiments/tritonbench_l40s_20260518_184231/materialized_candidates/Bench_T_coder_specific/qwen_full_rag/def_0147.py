import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def fused_l2normalize_and_distance_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_rows_x,
    n_cols,
    stride_x,
    stride_y,
    stride_out,
    eps_normalizer: float,
    eps_dist: float,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    X_BLOCK_PTR = tl.make_block_ptr(
        base=x_ptr,
        shape=(n_rows_x, n_cols),
        strides=(stride_x, 1),
        offsets=(row_idx, 0),
        block_shape=(BLOCK_SIZE, n_cols),
        order=(1, 0),
    )
    Y_BLOCK_PTR = tl.make_block_ptr(
        base=y_ptr,
        shape=(n_rows_x, n_cols),
        strides=(stride_y, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, n_cols),
        order=(1, 0),
    )
    OUT_BLOCK_PTR = tl.make_block_ptr(
        base=out_ptr,
        shape=(n_rows_x, n_cols),
        strides=(stride_out, 1),
        offsets=(row_idx, 0),
        block_shape=(BLOCK_SIZE, n_cols),
        order=(1, 0),
    )

    # Load data
    x = tl.load(X_BLOCK_PTR, boundary_check=(0,), padding_option="zero")
    x_2 = tl.where(x == x, x * x, 0)
    y = tl.load(Y_BLOCK_PTR, boundary_check=(0,), padding_option="zero")

    # Normalize
    mean_x_2 = tl.sum(x_2, axis=0) / n_cols
    inv_std_x = 1 / tl.sqrt(mean_x_2 + eps_normalizer)

    x_hat = x * inv_std_x

    # Distance
    x_minus_y = x_hat - y
    squared_dist = x_minus_y * x_minus_y
    dist = tl.sqrt(tl.sum(squared_dist, axis=0) + eps_dist) * inv_std_x

    # Write-back
    tl.store(OUT_BLOCK_PTR, dist.to(x.dtype), boundary_check=(0,))


def fused_pairwise_distance_normalize(
    x1: torch.Tensor,
    x2: torch.Tensor,
    p_norm: float = 2.0,
    eps_norm: float = 1e-12,
    eps_distance: float = 1e-6,
    keepdim: Optional[bool] = False,
) -> torch.Tensor:
    assert (
        x1.dim() > 1
    ), f"Input tensor must have at least 2 dimensions, got {x1.dim()} instead"
    assert x1.shape[-1] == x2.shape[
        -1
    ], "The last dimension of both tensors must match"
    assert all(
        [val >= 0 for val in [eps_norm, eps_distance]]
    ), "All epsilons must be greater than or equal to 0"

    rows, cols = x1.shape
    out = torch.empty(rows, device=x1.device, dtype=torch.float32)

    if p_norm != 2.0:
        x1 = x1.abs().pow(p_norm).mul(1.0 / p_norm)
        x2 = x2.abs().pow(p_norm).mul(1.0 / p_norm)
    else:
        x1 = x1.float()
        x2 = x2.float()

    if not keepdim:
        out = out.squeeze(dim=-1)

    BLOCK_SIZE = triton.next_power_of_2(cols)
    grid = lambda meta: (rows, 1, 1)
    fused_l2normalize_and_distance_kernel[grid](
        x1,
        x2,
        out,
        rows,
        cols,
        x1.stride(0),
        x2.stride(0),
        out.stride(0),
        eps_norm,
        eps_distance,
        BLOCK_SIZE,
    )

    return out
