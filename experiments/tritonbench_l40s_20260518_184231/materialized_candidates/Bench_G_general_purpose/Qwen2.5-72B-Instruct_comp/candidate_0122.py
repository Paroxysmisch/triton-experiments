import triton
import triton.language as tl

@triton.jit
def _rope_embedding(
    Q, Q_row_stride,
    cos, sin,
    cos_row_stride, sin_row_stride,
    seqlen, head_dim, n_heads,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute the head and sequence index
    head_idx = offsets // (seqlen * head_dim)
    seq_idx = (offsets % (seqlen * head_dim)) // head_dim
    dim_idx = offsets % head_dim

    # Check if the current block is within bounds
    mask = offsets < (seqlen * head_dim * n_heads)
    head_idx = tl.where(mask, head_idx, 0)
    seq_idx = tl.where(mask, seq_idx, 0)
    dim_idx = tl.where(mask, dim_idx, 0)

    # Load Q, cos, and sin
    Q_offsets = head_idx * Q_row_stride + seq_idx * head_dim + dim_idx
    Q_vals = tl.load(Q + Q_offsets, mask=mask, other=0.0)

    cos_offsets = seq_idx * cos_row_stride + dim_idx
    cos_vals = tl.load(cos + cos_offsets, mask=mask, other=0.0)

    sin_offsets = seq_idx * sin_row_stride + dim_idx
    sin_vals = tl.load(sin + sin_offsets, mask=mask, other=0.0)

    # Compute the RoPE transformation
    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_vals, -Q_vals)
    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)
    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated - head_dim // 2)
    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated + head_dim // 2)

    Q_rotated = tl.where(dim_idx < head_dim // 2, Q_rotated, Q_rotated * -1)
