import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _pairwise_distance_forward_kernel(
    x1_ptr, x2_ptr, out_ptr,
    N, D, p, eps,
    BLOCK_SIZE: tl.constexpr
):
    """
    Each program handles a range of rows in x1/x2 to compute:
       out[i] = ( sum_j( | x1[i, j] - x2[i, j] |^p ) + eps )^(1/p)
    for a given i, with j in [0, D-1].
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    end = tl.minimum(start + BLOCK_SIZE, N)

    # Loop over each row index from start to end
    i = start + tl.arange(0, BLOCK_SIZE)
    mask_i = i < end

    # We'll accumulate the intermediate sum of (|x1 - x2|^p).
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Stride in memory to access row elements
    # i * D gives the row offset for x1/x2.
    # We'll reduce across columns in multiple steps of BLOCK_SIZE or smaller chunk.
    for col_offset in range(0, D, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask_cols = cols < D

        # Read x1 and x2 elements
        # row offset is i * D, plus current col index
        # broadcast i for row, add cols to get final pointer offsets
        x1_idx = i * D + cols
        x2_idx = i * D + cols

        # Guard loads
        mask_load = mask_i & mask_cols
        x1_val = tl.load(x1_ptr + x1_idx, mask=mask_load, other=0.0)
        x2_val = tl.load(x2_ptr + x2_idx, mask=mask_load, other=0.0)

        # diff^p (general p might be fractional, so we use exponent)
        diff = x1_val - x2_val
        abs_diff = tl.abs(diff)
        diff_p = tl.power(abs_diff, p)
        acc += diff_p

    # Now we have sum of |x1 - x2|^p in acc, apply eps and do ^(1/p)
    acc = acc + eps
    acc = tl.power(acc, 1.0 / p)

    # Write out result
    tl.store(out_ptr + i, acc, mask=mask_i)


def fused_pairwise_distance_adaptive_avg_pool2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    output_size: int or tuple,
    p: float = 2.0,
    eps: float = 1e-6,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Applies adaptive average pooling to x1 and x2, then computes pairwise distance
    with norm p, adding eps to avoid division by zero. Optionally keeps the reduced dimension.
    """
    # 1) Adaptive Average Pooling
    x1_pooled = F.adaptive_avg_pool2d(x1, output_size=output_size)
    x2_pooled = F.adaptive_avg_pool2d(x2, output_size=output_size)

    # 2) Flatten all dimensions except the batch
    N = x1_pooled.shape[0]
    D = x1_pooled[0].numel()
    x1_flat = x1_pooled.view(N, D).contiguous()
    x2_flat = x2_pooled.view(N, D).contiguous()

    # Allocate output
    out = torch.empty(N, device=x1.device, dtype=x1.dtype)

    # Launch Triton kernel
    BLOCK_SIZE = 256
    grid = lambda META: ( (N + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'], )
    _pairwise_distance_forward_kernel[grid](
        x1_flat, x2_flat, out,
        N, D, p, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Optionally keep dimension
    if keepdim:
        out = out.view(-1, 1)

    return out
