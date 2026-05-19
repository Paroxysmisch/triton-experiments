import triton
import triton.language as tl
import torch

@triton.jit
def _rotary_kernel(
    Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
    Q_batch_stride, Q_head_stride, Q_seq_stride, Q_dmodel_stride,
    K_batch_stride, K_head_stride, K_seq_stride, K_dmodel_stride,
    Cos_seq_stride, Cos_dmodel_stride,
    Sin_seq_stride, Sin_dmodel_stride,
    batch_size, num_heads, seq_len, d_model,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_batch_head = tl.program_id(0)
    pid_seq = tl.program_id(1)

    # Calculate batch and head indices
    batch_head_idx = pid_batch_head
    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads

    # Calculate sequence index
    seq_idx = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

    # Bounds checking for batch, head, and sequence
    within_bounds = (batch_idx < batch_size) & (head_idx < num_heads) & (seq_idx < seq_len)

    # Calculate offsets for Q and K
    q_offset = (
        batch_idx * Q_batch_stride +
        head_idx * Q_head_stride +
        seq_idx[:, None] * Q_seq_stride +
        tl.arange(0, BLOCK_DMODEL)[None, :] * Q_dmodel_stride
    )
    k_offset = (
        batch_idx * K_batch_stride +
        head_idx * K_head_stride +
        seq_idx[:, None] * K_seq_stride +
        tl.arange(0, BLOCK_DMODEL)[None, :] * K_dmodel_stride
    )

    # Calculate offsets for Cos and Sin
    cos_offset = seq_idx[:, None] * Cos_seq_stride + tl.arange(0, BLOCK_DMODEL)[None, :] * Cos_dmodel_stride
    sin_offset = seq_idx[:, None] * Sin_seq_stride + tl.arange(0, BLOCK_DMODEL)[None, :] * Sin_dmodel_stride

    # Load Q, K, Cos, Sin with masking
    mask = within_bounds[:, None] & (tl.arange(0, BLOCK_DMODEL)[None, :] < d_model)
    q = tl.load(Q_ptr + q_offset, mask=mask, other=0.0)
    k = tl.load(K_ptr + k_offset, mask=mask, other=0.0)
    cos = tl.load(Cos_ptr + cos_offset, mask=mask, other=0.0)
    sin = tl.load(Sin_ptr + sin_offset, mask=mask, other=0.0)

    # Split into pairs (q0, q1) and (k0, k1)
    q0 = q[:, 0::2]
    q1 = q[:, 1::2]
    k0 = k[:, 0::2]
    k1 = k[:, 1::2]

    # Apply rotary embeddings
    cos = cos[:, 0::2]  # Use same cos for both components
    sin = sin[:, 0::2]
    q_rot0 = q0 * cos - q1 * sin
    q_rot1 = q0 * sin + q1 * cos
    k_rot0 = k0 * cos - k1 * sin
    k_rot1 = k0 * sin + k1 * cos

    # Interleave the rotated pairs
    q_rot = tl.zeros_like(q)
    q_rot = tl.where(tl.arange(0, BLOCK_DMODEL)[None, :] % 2 == 0, q_rot0, q_rot)
    q_rot = tl.where(tl.arange(0, BLOCK_DMODEL)[None, :] % 2 == 1, q_rot1, q_rot)
    k_rot = tl.zeros_like(k)
    k_rot = tl.where(tl.arange(0, BLOCK_DMODEL)[None, :] % 2 == 0, k_rot0, k_rot)
    k_rot = tl.where(tl.arange(0, BLOCK_DMODEL)[None, :] % 2 == 1, k_rot1, k_rot)

    # Store the results
    tl.store(Q_ptr + q_offset, q_rot, mask=mask)
    tl.store(K_ptr + k_offset, k_rot, mask=mask)
    
def rotary_emb_fwd(Q, K, Cos, Sin):
    assert Q.shape == K.shape, "Q and K must have the same shape"
    assert Cos.shape == Sin.shape, "Cos and Sin must have the same shape"
    batch_size, num_heads, seq_len, d_model = Q.shape

    # Determine block sizes
    BLOCK_HEAD = 1
    BLOCK_SEQ = 16
    BLOCK_DMODEL = 64  # Must be even and <= d_model

    # Compute grid dimensions
    grid = (batch_size * num_heads, triton.cdiv(seq_len, BLOCK_SEQ))

    # Calculate number of warps
    num_warps = 4 if d_model >= 128 else 2

    # Launch kernel
    _rotary_kernel[grid](
        Q, K, Cos, Sin,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        Cos.stride(0), Cos.stride(1),
        Sin.stride(0), Sin.stride(1),
        batch_size, num_heads, seq_len, d_model,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
    )
    return Q, K
