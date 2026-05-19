import torch
import triton
import triton.language as tl


@triton.jit
def _welford_update(mean, m2, weight, new_value, count):
    # Welford update for calculating mean and variance
    # see: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    delta = new_value - mean
    new_weight = weight + count
    new_mean = mean + delta * (count / new_weight)
    new_m2 = m2 + delta * (new_value - new_mean)
    return new_mean, new_m2, new_weight


@triton.jit
def triton_red_fused_native_layer_norm_0(
    x_ptr,        # primals_3: input (S, D)
    gamma_ptr,    # primals_1: scale
    beta_ptr,     # primals_2: bias
    out_stats,    # out_ptr0: buffer for mean, var, (optionally count)
    out_norm,     # out_ptr1: buffer for final normalized output
    S,            # number of rows
    D,            # number of columns
    RBLOCK,       # block size for columns
    eps,          # epsilon
    **META
):
    """
    Each program loads one row from x, does a Welford reduction to get
    mean and var, then writes normalized output to out_norm.
    out_stats[row * 3 + 0] = mean
    out_stats[row * 3 + 1] = var
    out_stats[row * 3 + 2] = count (for demonstration, typically D)
    """
    row_id = tl.program_id(0)
    # pointers to row start
    row_offset = row_id * D

    # -------------------------
    # Pass 1: Welford reduction
    # -------------------------
    mean_acc = tl.zeros([], dtype=tl.float32)
    m2_acc = tl.zeros([], dtype=tl.float32)
    weight_acc = tl.zeros([], dtype=tl.float32)

    # block-wise reduce the row
    # each iteration loads up to RBLOCK columns
    # and updates Welford accumulators
    col_start = 0
    while col_start < D:
        offs = col_start + tl.arange(0, RBLOCK)
        mask = offs < D
        x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0).to(tl.float32)

        # we treat each element as "count=1" in Welford update
        for i in range(RBLOCK):
            if mask[i]:
                mean_acc, m2_acc, weight_acc = _welford_update(
                    mean_acc,
                    m2_acc,
                    weight_acc,
                    x[i],
                    1.0
                )
        col_start += RBLOCK

    var_acc = m2_acc / weight_acc
    # store mean, var, and weight in out_stats
    tl.store(out_stats + row_id * 3 + 0, mean_acc)
    tl.store(out_stats + row_id * 3 + 1, var_acc)
    tl.store(out_stats + row_id * 3 + 2, weight_acc)

    # -------------------------
    # Pass 2: Apply normalization
    # -------------------------
    inv_std = 1
