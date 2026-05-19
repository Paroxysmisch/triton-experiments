import torch
import triton
import triton.language as tl

@triton.jit
def _fused_add_mul_groupnorm_kernel(
    x_ptr,           # X data
    y_ptr,           # Y data
    w_ptr,           # weight (gamma)
    b_ptr,           # bias (beta)
    out_ptr,         # output
    N,               # batch size
    C,               # #channels
    HW,              # height*width (or product of spatial dims)
    group_size,      # channels per group
    eps,             # epsilon
    BLOCK_SIZE: tl.constexpr
):
    """
    Each program (block) processes BLOCK_SIZE elements in the flattened spatial dimension
    for a single (n, c) pair. This kernel does the following per element of M:
        Z = X + Y
        M = Z * Y
        Then, partial (group-wide) sum for M to compute mean/var for group normalization.
        Final group norm uses w_ptr and b_ptr with shape (C,) and group_size dividing C evenly.
    """
    # Program ID ranges over total = N * C
    pid = tl.program_id(0)
    # n: batch index, c: channel index
    n = pid // C
    c = pid % C

    # Pointers offset
    # Flattened index: n * (C * HW) + c * HW
    x_off = n * C * HW + c * HW
    y_off = n * C * HW + c * HW
    o_off = n * C * HW + c * HW

    # Lane IDs
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < HW - (HW // BLOCK_SIZE) * BLOCK_SIZE  # extra boundary mask for partial blocks

    # Load X and Y, add them (Z = X+Y), multiply with Y (M = Z*Y)
    # then store partial M for group-norm.
    # Because we might not have enough parallelism or might not perfectly align,
    # we clamp the global index to avoid OOB reads:
    offset = x_off + idx
    offset_clamped = tl.where(idx < HW, offset, x_off + (HW - 1))
    X = tl.load(x_ptr + offset_clamped, mask=idx < HW)
    Y = tl.load(y_ptr + offset_clamped, mask=idx < HW)
    Z = X + Y
    M = Z * Y

    # We'll write M to out temporarily, then do group sums in another pass or approach.
    # For demonstration, store partial result (M) in out before normalizing. We'll
    # finalize normalization in a second pass or a reduce. 
    # In practice, you'd typically use shared memory or do multiple phases, but
    # for brevity here we'll store partial results in out, to do a group reduce in device code.

    tl.store(out_ptr + offset_clamped, M, mask=idx < HW)


@triton.jit
def _groupnorm_reduce_kernel(
    out_ptr,        # pointer to M data (from previous kernel)
    w_ptr,          # weight (gamma)
    b_ptr,          # bias (beta)
    final_ptr,      # final output pointer
    N, C, HW,
    group_size,     # channels per group
    eps,            # epsilon
    BLOCK_SIZE: tl.constexpr
):
    """
    This kernel completes the group normalization using partial data in out_ptr.
    We'll compute group means/vars for each group, then do final affine transform.
    For simplicity, we do one block per (n, c), same as above, but
    we reduce over the group using a for-loop. This is not fully optimized,
    but demonstrates the concept in a single pass for each element with group stats.
    """
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    # Find group index and group offset
    group_idx = c // group_size
    group_c_start = group_idx * group_size
    # In-group relative channel
    offset_in_group = c - group_c_start

    # We'll compute group mean/var by iterating over group channels
    # and across the spatial dimension for each.
    sum_ = tl.zeros([1], dtype=tl.float32)
    sq_sum_ = tl.zeros([1], dtype=tl.float32)
    group_channel_range = tl.arange(0, group_size)
    # Accumulate sums for group mean/var
    # naive approach: for all channels in group, for all HW elements, load and accumulate
    # We'll skip parallelization intricacies for brevity.

    # Full length for group channels is group_size * HW
    # We'll do a simple for-loop in device code (not recommended for performance, but simpler):
    base_nHW = n * C * HW
    group_start_off = base_nHW + group_c_start * HW

    for ch in range(group_size):
        ch_off = group_start_off + ch * HW
        # Accumulate for all HW
        # We'll process BLOCK_SIZE at a time
        for block_start in range(0, HW, BLOCK_SIZE):
            idx = tl.arange(0, BLOCK_SIZE)
            valid = (block_start + idx) < HW
            off_clamped = ch_off + block_start + tl.where(valid, idx, BLOCK_SIZE - 1)
            M_chunk = tl.load(out_ptr + off_clamped, mask=valid)
            sum_ += tl.sum(M_chunk, where=valid)
            sq_sum_ += tl.sum(M_chunk * M_chunk, where=valid)

    group_elems = group_size * HW
    mean = sum_ / group_elems
    var = sq_sum_ / group_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Finally, we load M for the current channel c, do normalization, multiply by gamma, add beta.
    # Then store in final_ptr.
    # We'll do another pass over HW in BLOCK_SIZE chunks.
    base_ch_off = base_nHW + c * HW
    gamma = tl.load(w_ptr + c)
    beta = tl.load(b_ptr + c)

    for block_start in range(0, HW, BLOCK_SIZE):
        idx = tl.arange(0, BLOCK_SIZE)
        valid = (block_start + idx) < HW
        off_clamped = base_ch_off + block_start + tl.where(valid, idx, BLOCK_SIZE - 1)

        M_chunk = tl.load(out_ptr + off_clamped, mask=valid)
        normed = (M_chunk - mean) * inv_std
        out_val = normed * gamma + beta
        tl.store(final_ptr + off_clamped, out_val, mask=valid)


def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    """
    fused_add_mul_groupnorm(input1, input2, weight
