import triton
import triton.language as tl
import torch

@triton.jit
def rms_matmul_rbe(
    x_ptr,         # (B*M, K)
    w_ptr,         # (K, N) (transposed weight)
    rms_w_ptr,     # (K,) RMS weight
    y_ptr,         # (B*M, N) output
    B, M, N, K,    # batch size, seq length, output dim, input dim
    stride_xk,     # distance (in elements) between consecutive K elements in x
    stride_wn,     # distance (in elements) between consecutive N elements in w
    stride_y,      # distance (in elements) to next row in y
    USE_ROTARY: tl.constexpr, # whether to apply rotary embeddings
    THETA: tl.float32,        # angle for rotary
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """
    Each program processes a [BLOCK_SIZE_M x BLOCK_SIZE_N] tile of the output.
    The input x is arranged as (B*M, K) so that each row is a sequence token (possibly across batches).
    The weight w is (K, N).
    RMS normalization is applied to each row of x before multiplication, using rms_w_ptr as gain.
    If USE_ROTARY is True, apply a simple rotary embedding transform on the (already normalized) x slice.
    """
    # Program IDs - we map program_id(0) to the row dimension, program_id(1) to the col dimension.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute range of rows and cols processed by this program
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N

    # Create 2D ranges
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    # For partial accesses
    mask_m = row_start + offs_m < B * M
    mask_n = col_start + offs_n < N

    # ----------------
    # Load and RMS-normalize the slice of x for the block [row_start : row_start+BLOCK_SIZE_M]
    # We load the entire row K dimension in steps of BLOCK_SIZE_K, compute sum of squares, then normalize.
    x_sumsq = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    # We'll accumulate the sum of squares in chunks
    for k_block_start in range(0, K, BLOCK_SIZE_K):
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        kk = k_block_start + offs_k
        k_mask = kk < K
        # We gather each row in the tile from X
        # shape [BLOCK_SIZE_M, BLOCK_SIZE_K]
        x_row_ptrs = x_ptr + (row_start + offs_m[:, None]) * stride_xk + kk[None, :]
        x_vals = tl.load(x_row_ptrs, mask=(mask_m[:, None] & k_mask[None, :]), other=0.0)
        # Accumulate sum of squares
        x_sumsq += tl.sum(x_vals * x_vals, 1)

    # Now each of the BLOCK_SIZE_M rows in x_sumsq is the sum of squares across K
    # Compute RMS and load gain from rms_w_ptr
    # We'll do the final pass to read x again, normalize, and then multiply w in the same loop
    # so let's prepare partial accum for output
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # RMS
    # Add small epsilon for numerical stability
    eps = 1e-9
    rms_vals = tl.sqrt(x_sumsq / tl.float32(K) + eps)

    # Repeat pass over K in blocks, multiply with W, accumulate to acc.
    for k_block_start in range(0, K, BLOCK_SIZE_K):
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        kk = k_block_start + offs_k
        k_mask = kk < K

        # Load slice of x
        x_row_ptrs = x_ptr + (row_start + offs_m[:, None]) * stride_xk + kk[None, :]
        x_vals = tl.load(x_row_ptrs, mask=(mask_m[:, None] & k_mask[None, :]), other=0.0)

        # Normalize x_vals
        # Load RMS weight (gain) once per K element
        rms_gain_ptrs = rms_w_ptr + kk
        rms_gain = tl.load(rms_gain_ptrs, mask=k_mask, other=1.0)
        # x_norm[i, j] = x_vals[i, j] / rms_vals[i] * rms_gain[j]
        # broadcast along row-dim for the 1 / rms_vals, along col-dim for rms_gain
        # shape is [BLOCK_SIZE_M, BLOCK_SIZE_K]
        x_normed = x_vals / rms_vals[:, None]
        x_normed = x_normed * rms_gain[None, :]

        # (Optional) Rotary Embeddings (very simplified example)
        if USE_ROTARY:
            # We'll just rotate the real/even and imaginary/odd halves inside x_normed
            # for demonstration. (Assume K is even, or at least the block is small.)
            # We'll interpret pairs (2i, 2i+1) as cos/sin.
            idx_rot = k_block_start + offs_k
            cos_val = tl.cos(THETA * idx_rot, out_dtype=tl.float32)
            sin_val = tl.sin(THETA * idx_rot, out_dtype=tl.float32)
            # rotate: x -> x*cos + x_swap*sin
            # We do a simplistic approach: we rotate pairs
            # If an index is even, we rotate with next odd, etc.
            even_mask = (idx_rot % 2) == 0
            odd_mask = ~even_mask
            x_even = tl.where(even_mask[None, :], x_normed, tl.zeros_like(x_normed))
            x_odd = tl.where(odd_mask[None, :], x_normed, tl.zeros_like(x_normed))
            x_rot_even = x_even * cos_val[None, :] - x_odd * sin_val[None, :]
            x_rot_odd = x_even * sin_val[None, :] + x_odd * cos_val[None,
