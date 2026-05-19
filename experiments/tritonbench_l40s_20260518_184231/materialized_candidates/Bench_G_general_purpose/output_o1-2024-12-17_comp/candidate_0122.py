import torch
import triton
import triton.language as tl

@triton.jit
def _rope_embedding(
    Q_ptr,
    Q_row_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    seqlen,
    head_dim,
    n_heads,
    BLOCK_SIZE: tl.constexpr,
    BACKWARD_PASS: tl.constexpr
):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_id = tl.program_id(1) * BLOCK_SIZE + col_offsets

    # Guard: no out-of-bounds
    mask = col_id < head_dim
    col_id = tl.where(mask, col_id, head_dim - 1)

    # Each row_id corresponds to a single sequence element + head index
    # row_id = bid * (seqlen*n_heads) + i => batch offset can be integrated here if needed
    seq_idx = row_id // n_heads
    head_idx = row_id % n_heads

    # Offsets in memory
    Q_offset = row_id * Q_row_stride + col_id
    cos_offset = seq_idx * cos_row_stride + col_id
    sin_offset = seq_idx * sin_row_stride + col_id

    # Load Q, cos, sin
    Q = tl.load(Q_ptr + Q_offset, mask=mask, other=0.0)
    c = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0)
    s = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0)

    half_dim = head_dim // 2
    # Indices for the second half of Q for rotate
    rot_id = col_id + half_dim
    rot_mask = (col_id < half_dim) & (rot_id < head_dim)

    # Compute rotated Q
    rot_val = tl.zeros_like(Q)
    rot_val = tl.where(
        rot_mask,
        tl.load(Q_ptr + row_id * Q_row_stride + rot_id, mask=rot_mask, other=0.0),
        rot_val
    )

    # Forward or Backward pass
    # forward: Q_out = Q * cos + rotate_half(Q) * sin
    # backward: Q_out = Q * cos - rotate_half(Q) * sin
    if BACKWARD_PASS == 1:
        Q_out = Q * c - rot_val * s
    else:
        Q_out = Q * c + rot_val * s

    tl.store(Q_ptr + Q_offset, Q_out, mask=mask)

class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        batch_size, seqlen, n_heads, head_dim = Q.shape
        Q_reshaped = Q.view(-1, head_dim)
        cos_reshaped = cos.view(-1, head_dim)
        sin_reshaped = sin.view(-1, head_dim)

        Q_ptr = Q_reshaped.data_ptr()
        cos_ptr = cos_reshaped.data_ptr()
        sin_ptr = sin_reshaped.data_ptr()

        grid = lambda meta: (Q_reshaped.shape[0], triton.cdiv(head_dim, meta['BLOCK_SIZE']))
        BLOCK_SIZE = 128

        _rope_embedding[grid](
            Q_ptr,
            Q_reshaped.stride(0),
            cos_ptr,
            cos_reshaped.stride(0),
            sin_ptr,
            sin_reshaped.stride(0),
            seqlen,
            head_dim,
            n_heads,
            BLOCK_SIZE=BLOCK_SIZE,
            BACKWARD_PASS=0
        )

        ctx.save_for_backward(Q, cos, sin)
        return Q_reshaped.view(batch_size, seqlen, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dQ):
        Q, cos, sin = ctx.saved_tensors
        batch_size, seqlen, n_heads, head_dim = Q.shape

        dQ_reshaped = dQ.view(-1, head_dim)
        cos_reshaped = cos.view(-1, head_dim)
        sin_reshaped = sin.view(-1, head_dim)

        dQ_ptr = dQ_reshaped.data_ptr()
        cos_ptr = cos_reshaped.data_ptr()
        sin_ptr = sin_reshaped.data_ptr()

        grid = lambda meta: (dQ_reshaped.shape[0], triton.cdiv(head_dim, meta['BLOCK_SIZE']))
        BLOCK_SIZE = 128

        _rope_embedding[grid](
            dQ_ptr,
            dQ_reshaped.stride(0),
            cos_ptr,
            cos_reshaped.stride(0),
            sin_ptr,
            sin_reshaped.stride(0),
            seqlen,
            head_dim,
            n_heads,
            BLOCK_SIZE=BLOCK_SIZE,
            BACKWARD_PASS=1
        )

        return dQ_reshaped.view(batch_size, seqlen, n_heads, head_dim), None, None

def fast_rope_embedding(Q, K, cos, sin):
    Q_out = Fast_RoPE_Embedding.apply(Q.transpose(1, 2), cos, sin).transpose(1, 2)
    K_out = Fast_RoPE_Embedding.apply(K.transpose(1, 2), cos, sin).transpose(1, 2)
    return Q_out, K_out
