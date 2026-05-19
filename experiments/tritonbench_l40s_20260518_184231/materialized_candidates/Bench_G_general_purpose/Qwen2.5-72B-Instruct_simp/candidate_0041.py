import triton
import triton.language as tl

@triton.jit
def _triton_rope(q, k, cos, sin, q_batch_stride, q_head_stride, q_seq_stride, q_embed_stride, 
                 k_batch_stride, k_head_stride, k_seq_stride, k_embed_stride, 
                 cos_seq_stride, cos_embed_stride, sin_seq_stride, sin_embed_stride, 
                 seq_len, embed_dim, BLOCK_SIZE: 
    pid = tl.program_id(axis=0)
    batch_id = pid // (seq_len * embed_dim)
    seq_id = (pid % (seq_len * embed_dim)) // embed_dim
    embed_id = (pid % (seq_len * embed_dim)) % embed_dim

    q_offset = batch_id * q_batch_stride + seq_id * q_seq_stride + embed_id * q_embed_stride
    k_offset = batch_id * k_batch_stride + seq_id * k_seq_stride + embed_id * k_embed_stride
    cos_offset = seq_id * cos_seq_stride + embed_id * cos_embed_stride
    sin_offset = seq_id * sin_seq_stride + embed_id * sin_embed_stride

    q_val = tl.load(q + q_offset)
    k_val = tl.load(k + k_offset)
    cos_val = tl.load(cos + cos_offset)
    sin_val = tl.load(sin + sin_offset)

    q_new = q_val * cos_val - k_val * sin_val
    k_new = k_val * cos_val + q_val * sin_val

    tl.store(q + q_offset, q_new)
    tl.store(k + k_offset, k_new)

import torch
import triton
import triton.language as tl

def rope_forward(q, k, cos, sin):
    assert q.shape == k.shape, "Query and Key tensors must have the same shape"
    assert q.shape[-1] == cos.shape[-1] == sin.shape[-1], "Embedding dimension mismatch"
    assert q.shape[-2] == cos.shape[-2] == sin.shape[-2], "Sequence length mismatch"

    batch_size, num_heads, seq_len, embed_dim = q.shape

    # Rearrange dimensions to fit hardware constraints
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # Calculate padding sizes
    block_size = 128  # Example block size, can be adjusted
    padded_seq_len = (seq_len + block_size - 1) // block_size * block_size
    padded_embed_dim = (embed_dim + block_size - 1) // block_size * block_size

    # Pad tensors if necessary
    if padded_seq_len > seq_len or padded_embed_dim > embed_dim:
        q = torch.nn.functional.pad(q, (0, padded_embed_dim - embed_dim, 0, padded_seq_len - seq_len))
        k = torch.nn.functional.pad(k, (0, padded_embed_dim - embed_dim, 0, padded_seq_len - seq_len))
        cos = torch.nn.functional.pad(cos, (0, padded_embed_dim - embed_dim, 0, padded_seq_len - seq_len))
        sin = torch.nn.functional.pad(sin, (0, padded_embed_dim - embed_dim, 0, padded_seq_len - seq_len))

    # Launch the kernel
    grid = (batch_size * padded_seq_len * padded_embed_dim, )
    _triton_rope[grid](
        q, k, cos, sin,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0), cos.stride(1), cos.stride(2), cos.stride(3),
        sin.stride(0), sin.stride(1), sin.stride(2), sin.stride(3),
        padded_seq_len, padded_embed_dim, block_size
    )

    # Remove padding if necessary
    q = q[:, :, :seq_len, :embed_dim]
    k = k[:, :, :seq_len, :embed_dim]
    cos = cos[:, :, :seq_len, :embed_dim]
    sin = sin[:, :, :seq_len, :embed_dim]

    return q, k, cos, sin
