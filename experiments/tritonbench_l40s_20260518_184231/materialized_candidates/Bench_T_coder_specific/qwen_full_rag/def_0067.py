import torch
import triton
import triton.language as tl
from torch import Tensor
from .utils import calculate_settings_2d

@triton.jit
def _fwd__pairwise_distance(x_ptr, y_ptr, d_ptr, M, N, D, rM, rN, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    """Kernel for computing the pairwise distance upper triangular part only.

    ..Note: Better performance when XBLOCK is slightly smaller than the actual
    block size (e.g., 1024 instead of 1024) and is a divisor of 2048.

    Parameters
    ----------
    x_ptr : Tensor
        Input tensor of shape (M, D).
    y_ptr : Tensor
        Input tensor of shape (N, D).
    d_ptr : Tensor
        Output distance tensor of shape (rM, rN).
    M : int
        Number of rows in x_ptr.
    N : int
        Number of rows in y_ptr.
    D : int
        Number of columns in x_ptr and y_ptr.
    rM : int
        Chunk size along the M dimension.
    rN : int
        Chunk size along the N dimension.
    XBLOCK : tl.constexpr
        Block size along the M dimension.
    RBLOCK : tl.constexpr
        Block size along the N dimension.
    """
    # Program ids for each loop
    pid_x = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)

    # Calculate the offsets for each loop
    x_start = pid_x * XBLOCK
    r_start = pid_r * RBLOCK

    # Create masks for valid indices
    x_mask = x_start < M
    r_mask = r_start < N

    # Load x and y blocks with masking
    x_block_ptr = x_ptr + (x_start + tl.arange(0, XBLOCK)).to(tl.int64)[:, None] * D
    y_block_ptr = y_ptr + (r_start + tl.arange(0, RBLOCK)).to(tl.int64)[None, :] * D
    x = tl.load(x_block_ptr, mask=(x_mask)[:, None], other=0.0)
    y = tl.load(y_block_ptr, mask=(r_mask)[None, :], other=0.0)

    # Compute pairwise distance and store results
    for i in range(0, D, 16):
        i = i.to(tl.int64)
        x_ = tl.load(x_block_ptr + i, mask=(x_mask)[:, None] & (i < D - i), other=0.0)
        y_ = tl.load(y_block_ptr + i, mask=(r_mask)[None, :] & (i < D - i), other=0.0)
        d = x_ - y_
        d = tl.math.abs(d)
        d2 = tl.math.pow(d, 2)
        d_ = tl.sqrt(tl.math.sum(d2, 0))
        off_ = (x_start + tl.arange(0, XBLOCK)) * N + (r_start + tl.arange(0, RBLOCK))
        d_ptr = d_ptr + off_[None, :]
        tl.store(d_ptr, d_, mask=(x_mask)[:, None] & (r_mask)[None, :])


@triton.jit
def _bwd__pairwise_distance(
    x_ptr, y_ptr, dd_ptr, dx_ptr, dy_ptr, M, N, D, rM, rN, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    """Kernel for computing the pairwise gradients upper triangular part only.

    ..Note: Better performance when XBLOCK is slightly smaller than the actual
    block size (e.g., 1024 instead of 1024) and is a divisor of 2048.

    Parameters
    ----------
    x_ptr : Tensor
        Input tensor of shape (M, D).
    y_ptr : Tensor
        Input tensor of shape (N, D).
    dd_ptr : Tensor
        Input gradient tensor of shape (rM, rN).
    dx_ptr : Tensor
        Output gradient tensor for x of shape (M, D).
    dy_ptr : Tensor
        Output gradient tensor for y of shape (N, D).
    M : int
        Number of rows in x_ptr.
    N : int
        Number of rows in y_ptr.
    D : int
        Number of columns in x_ptr and y_ptr.
    rM : int
        Chunk size along the M dimension.
    rN : int
        Chunk size along the N dimension.
    XBLOCK : tl.constexpr
        Block size along the M dimension.
    RBLOCK : tl.constexpr
        Block size along the N dimension.
    """
    # Program ids for each loop
    pid_x = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)
    # Calculate the offsets for each loop
    x_start = pid_x * XBLOCK
    r_start = pid_r * RBLOCK
    # Create masks for valid indices
    x_mask = x_start < M
    r_mask = r_start < N
    # Load x and y blocks with masking
    x_block_ptr = x_ptr + (x_start + tl.arange(0, XBLOCK)).to(tl.int64)[:, None] * D
    y_block_ptr = y_ptr + (r_start + tl.arange(0, RBLOCK)).to(tl.int64)[None, :] * D
    x = tl.load(x_block_ptr, mask=(x_mask)[:, None], other=0.0)
    y = tl.load(y_block_ptr, mask=(r_mask)[None, :], other=0.0)
    # Load dd blocks with masking
    dd_block_ptr = dd_ptr + (x_start + tl.arange(0, XBLOCK)[:, None]).to(tl.int64) * N + (
        r_start + tl.arange(0, RBLOCK)[None, :]
    )
    dd = tl.load(dd_block_ptr, mask=(x_mask)[:, None] & (r_mask)[None, :], other=0.0)
    # Compute pairwise gradients and store results
    for i in range(0, D, 16):
        i = i.to(tl.int64)
        x_ = tl.load(x_block_ptr + i, mask=(x_mask)[:, None] & (i < D - i), other=0.0)
        y_ = tl.load(y_block_ptr + i, mask=(r_mask)[None, :] & (i < D - i), other=0.0)
        dd_ = tl.load(dd_block_ptr + i, mask=(x_mask)[:, None] & (r_mask)[None, :], other=0.0)
        c = dd_ / (y_ - x_)
        dx_ = -dd_ * (y_ - x_)
        dx_ = tl.where(c <= 0, dx_, dx_ * (-c) * ((1 - c) + c * (y_ - x_) / (y_ - x_)))
        dx = tl.math.abs(dx_)
        dx = tl.math.pow(dx, 2)
        dx = tl.sqrt(tl.math.sum(dx, 0))
        dy = dx * (y_ - x_) / (dx + 1e-6)
        off_x_ = (x_start + tl.arange(0, XBLOCK))
        off_y_ = (r_start + tl.arange(0, RBLOCK))
        dx_ptr_ = dx_ptr + off_x_[:, None] * D + i
        dy_ptr_ = dy_ptr + off_y_[None, :] * D + i
        tl.store(dx_ptr_, dx, mask=(x_mask)[:, None] & (i < D - i))
        tl.store(dy_ptr_, dy, mask=(r_mask)[None, :] & (i < D - i))


class PairWiseDistanceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y, output_size, p, eps):
        if type(output_size) == int:
            output_size = (output_size, output_size)
        if len(output_size) != 2:
            raise ValueError(f"Invalid output size: {output_size}")
        output_size_ = (x.shape[0], y.shape[0])
        if output_size != output_size_:
            x = tl.F.adaptive_avg_pool2d(x.unsqueeze(0).unsqueeze(0), output_size_).squeeze()
            y = tl.F.adaptive_avg_pool2d(y.unsqueeze(0).unsqueeze(0), output_size_).squeeze()
        else:
            pass
        B, L, C = x.shape
        x = x.reshape(-1, C)
        y = y.reshape(-1, C)
        M, D = x.shape
        N, _ = y.shape
        rM, rN = triton.cdiv(M, 512), triton.cdiv(N, 512)
        num_stages = 4 if D <= 2048 else 3
        num_warps = 4
        d = torch.empty((rM, rN), dtype=x.dtype, device=x.device)
        _fwd__pairwise_distance[(rM, rN)](
            x,
            y,
            d,
            M,
            N,
            D,
            rM,
            rN,
            num_warps=num_warps,
            num_stages=num_stages,
