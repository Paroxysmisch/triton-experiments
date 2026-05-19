import triton
import triton.language as tl
import torch

@triton.jit
def _rotary_kernel(
    Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    n_heads, seq_len, d_model,
    BLOCK_HEAD: tl.constexpr, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Program ID for head and sequence dimensions
    pid_head = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    # Calculate offsets
    head_range = pid_head * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    seq_range = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    dim_range = tl.arange(0, BLOCK_DMODEL)
    
    # Create masks for boundary checking
    head_mask = head_range < n_heads
    seq_mask = seq_range < seq_len
    
    # Combine masks
    mask = head_mask[:, None] & seq_mask[None, :]
    
    # Calculate base offsets for Q and K
    q_offset = (head_range[:, None, None] * stride_q_h + 
                seq_range[None, :, None] * stride_q_s +
                dim_range[None, None, :] * stride_q_d)
    k_offset = (head_range[:, None, None] * stride_k_h + 
                seq_range[None, :, None] * stride_k_s +
                dim_range[None, None, :] * stride_k_d)
    
    # Load Q and K values
    q = tl.load(Q_ptr + q_offset, mask=mask[:, :, None])
    k = tl.load(K_ptr + k_offset, mask=mask[:, :, None])
    
    # Load Cos and Sin values
    cos_offset = seq_range[None, :, None] * stride_cos_s + dim_range[None, None, :] * stride_cos_d
    sin_offset = seq_range[None, :, None] * stride_sin_s + dim_range[None, None, :] * stride_sin_d
    
    cos = tl.load(Cos_ptr + cos_offset, mask=seq_mask[:, None])
    sin = tl.load(Sin_ptr + sin_offset, mask=seq_mask[:, None])
    
    # Apply rotary embeddings
    # For even indices
    q_even = q[..., ::2]
    q_odd = q[..., 1::2]
    k_even = k[..., ::2]
    k_odd = k[..., 1::2]
    
    cos_even = cos[..., ::2]
    sin_even = sin[..., ::2]
    
    # Compute rotated values
    q_out_even = q_even * cos_even - q_odd * sin_even
    q_out_odd = q_odd * cos_even + q_even * sin_even
    k_out_even = k_even * cos_even - k_odd * sin_even
    k_out_odd = k_odd * cos_even + k_even * sin_even
    
    # Interleave results back
    q_out = tl.where(dim_range % 2 == 0, q_out_even, q_out_odd)
    k_out = tl.where(dim_range % 2 == 0, k_out_even, k_out_odd)
    
    # Store results
    tl.store(Q_ptr + q_offset, q_out, mask=mask[:, :, None])
    tl.store(K_ptr + k_offset, k_out, mask=mask[:, :, None])

def rotary_emb_fwd(q, k, cos, sin):
    """
    Apply rotary embeddings to Q and K tensors
    
    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, d_model)
        k: Key tensor of shape (batch, n_heads, seq_len, d_model)
        cos: Cosine tensor of shape (seq_len, d_model)
        sin: Sine tensor of shape (seq_len, d_model)
    """
    batch, n_heads, seq_len, d_model = q.shape
    assert k.shape == q.shape
    assert cos.shape == (seq_len, d_model)
    assert sin.shape == (seq_len, d_model)
    
    # Define block sizes
    BLOCK_HEAD = triton.next_power_of_2(min(n_heads, 32))
    BLOCK_SEQ = triton.next_power_of_2(min(seq_len, 128))
    BLOCK_DMODEL = d_model
    
    # Calculate grid dimensions
    grid = (triton.cdiv(n_heads, BLOCK_HEAD), triton.cdiv(seq_len, BLOCK_SEQ))
    
    # Calculate number of warps based on head dimension
    num_warps = 4 if n_heads >= 256 else 2
    
    # Launch kernel
    _rotary_kernel[grid](
        q, k, cos, sin,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        n_heads, seq_len, d_model,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps
    )
    
    return q, k
