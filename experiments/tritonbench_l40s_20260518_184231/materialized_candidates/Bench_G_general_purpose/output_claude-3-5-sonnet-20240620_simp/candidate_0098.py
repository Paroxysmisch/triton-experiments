import torch
import triton
import triton.language as tl

@triton.jit
def _rotary_kernel(
    # Pointers to matrices
    Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
    # Matrix strides
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_cos, stride_sin,
    # Metadata
    max_total_len, HEAD_Q, HEAD_K,
    # Block sizes (compile-time constants)
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Program ID
    pid_head = tl.program_id(0)
    pid_seq = tl.program_id(1)

    # Initialize offsets
    head_range = pid_head * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    seq_range = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    dim_range = tl.arange(0, BLOCK_DMODEL)

    # Create masks for bounds checking
    head_mask = head_range < HEAD_Q
    seq_mask = seq_range < max_total_len

    # Compute Q offsets
    q_offs = (head_range[:, None, None] * stride_qh +
              seq_range[None, :, None] * stride_qm +
              dim_range[None, None, :])
    
    # Compute K offsets
    k_offs = (head_range[:, None, None] * stride_kh +
              seq_range[None, :, None] * stride_kn +
              dim_range[None, None, :])

    # Load Q and K
    q = tl.load(Q_ptr + q_offs, mask=head_mask[:, None, None] & seq_mask[None, :, None])
    k = tl.load(K_ptr + k_offs, mask=head_mask[:, None, None] & seq_mask[None, :, None])

    # Load Cos and Sin
    cos = tl.load(Cos_ptr + seq_range[None, :] * stride_cos + dim_range[:, None],
                 mask=seq_mask[None, :])
    sin = tl.load(Sin_ptr + seq_range[None, :] * stride_sin + dim_range[:, None],
                 mask=seq_mask[None, :])

    # Apply rotary transformation
    # For even indices
    q_even = q[..., ::2]
    q_odd = q[..., 1::2]
    k_even = k[..., ::2]
    k_odd = k[..., 1::2]

    # Compute rotary transformation
    q_out_even = q_even * cos[None, :, :] - q_odd * sin[None, :, :]
    q_out_odd = q_odd * cos[None, :, :] + q_even * sin[None, :, :]
    k_out_even = k_even * cos[None, :, :] - k_odd * sin[None, :, :]
    k_out_odd = k_odd * cos[None, :, :] + k_even * sin[None, :, :]

    # Interleave results back
    q_out = tl.where(dim_range % 2 == 0, q_out_even, q_out_odd)
    k_out = tl.where(dim_range % 2 == 0, k_out_even, k_out_odd)

    # Store results
    tl.store(Q_ptr + q_offs, q_out, mask=head_mask[:, None, None] & seq_mask[None, :, None])
    tl.store(K_ptr + k_offs, k_out, mask=head_mask[:, None, None] & seq_mask[None, :, None])

def rotary_emb_fwd(q, k, cos, sin):
    """
    Apply rotary embeddings to Q and K tensors.
    
    Args:
        q: Query tensor of shape [batch, head_q, seq_len, dim]
        k: Key tensor of shape [batch, head_k, seq_len, dim]
        cos: Cosine embeddings of shape [seq_len, dim//2]
        sin: Sine embeddings of shape [seq_len, dim//2]
    """
    batch, head_q, seq_len, dim = q.shape
    _, head_k, _, _ = k.shape

    # Compute strides
    stride_qb = q.stride(0)
    stride_qh = q.stride(1)
    stride_qm = q.stride(2)
    stride_kb = k.stride(0)
    stride_kh = k.stride(1)
    stride_kn = k.stride(2)
    stride_cos = cos.stride(0)
    stride_sin = sin.stride(0)

    # Define block sizes
    BLOCK_HEAD = 4
    BLOCK_SEQ = 64
    BLOCK_DMODEL = dim

    # Compute grid size
    grid = (triton.cdiv(head_q, BLOCK_HEAD), triton.cdiv(seq_len, BLOCK_SEQ))
    
    # Determine number of warps based on dimension size
    num_warps = 4 if dim <= 64 else 8

    # Launch kernel
    _rotary_kernel[grid](
        q, k, cos, sin,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_cos, stride_sin,
        seq_len, head_q, head_k,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps
    )

    return q, k
