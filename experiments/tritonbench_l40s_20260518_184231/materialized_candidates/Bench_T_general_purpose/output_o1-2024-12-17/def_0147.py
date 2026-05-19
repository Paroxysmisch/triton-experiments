import torch
import triton
import triton.language as tl


@triton.jit
def _pairwise_distance_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    stride_x1_m,
    stride_x1_d,
    stride_x2_m,
    stride_x2_d,
    stride_out_m,
    stride_out_n,
    M,  # number of rows in x1
    N,  # number of rows in x2
    D,  # dimension over which distance is computed
    eps_distance,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    """
    Computes pairwise distances between blocks of x1 and x2.
    Each warp processes a block of size (BLOCK_M x BLOCK_N).
    """
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)

    # Compute current row/col start
    m_start = m_block * BLOCK_M
    n_start = n_block * BLOCK_N

    # Create coordinate for sub-block
    rm = m_start + tl.arange(0, BLOCK_M)
    rn = n_start + tl.arange(0, BLOCK_N)

    # Create masks to guard memory accesses
    rm_mask = rm < M
    rn_mask = rn < N

    # We will compute the pairwise distances for coordinates [rm, rn].
    # Initialize an accumulator for each pair (partial sum of squares).
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over dimension D in chunks of BLOCK_D
    # We accumulate partial sums for difference^2.
    for d_offset in range(0, D, BLOCK_D):
        # Offsets for loading in the d-dimension
        d_idx = d_offset + tl.arange(0, BLOCK_D)
        d_mask = d_idx < D

        # Load x1 and x2 blocks
        # shape [BLOCK_M, BLOCK_D]
        x1_vals = tl.where(
            (rm_mask[:, None] & d_mask[None, :]),
            tl.load(x1_ptr + rm[:, None] * stride_x1_m + d_idx[None, :] * stride_x1_d),
            0.0
        )
        # shape [BLOCK_N, BLOCK_D]
        x2_vals = tl.where(
            (rn_mask[:, None] & d_mask[None, :]),
            tl.load(x2_ptr + rn[:, None] * stride_x2_m + d_idx[None, :] * stride_x2_d),
            0.0
        )

        # Expand dims for broadcasting difference^2
        diff = x1_vals[:, None, :] - x2_vals[None, :, :]
        acc += tl.sum(diff * diff, axis=2)

    # Now compute final distance = sqrt(acc + eps_distance)
    dist = tl.sqrt(acc + eps_distance)

    # Write results back
    # We store dist at out[rm, rn].
    # Each index is only valid if rm < M and rn < N.
    # shape [BLOCK_M, BLOCK_N]
    mask_mn = (rm_mask[:, None] & rn_mask[None, :])
    tl.store(
        out_ptr + rm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
        dist,
        mask=mask_mn
    )


def fused_pairwise_distance_normalize(
    x1: torch.Tensor,
    x2: torch.Tensor,
    p_norm: float = 2.0,
    eps_norm: float = 1e-12,
    eps_distance: float = 1e-6,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Normalizes x1 and x2 along their last dimension using p_norm, then computes
    pairwise distances between all rows of x1 and x2 using an L2 distance.
    Args:
        x1 (torch.Tensor): First input tensor, shape (..., D).
        x2 (torch.Tensor): Second input tensor, shape (..., D).
        p_norm (float, optional): Norm exponent used for normalization. Default: 2.0.
        eps_norm (float, optional): Epsilon to avoid division by zero during normalization. Default: 1e-12.
        eps_distance (float, optional): Epsilon to avoid zero distance. Default: 1e-6.
        keepdim (bool, optional): If True, retains an extra dimension in the output. Default: False.
    Returns:
        torch.Tensor: Pairwise distance matrix of shape (x1.size(0), x2.size(0)) or (x1.size(0
