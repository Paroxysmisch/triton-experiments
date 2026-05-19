import triton
import triton.language as tl
import torch

@triton.jit
def _rotary_kernel(
    # Pointers to input/output tensors
    Q_ptr, K_ptr,
    Cos_ptr, Sin_ptr,
    # Tensor strides
    stride_q_batch, stride_q_head, stride_q_seq, stride_q_model,
    stride_k_batch, stride_k_head, stride_k_seq, stride_k_model,
    stride_cos_seq, stride_cos_model,
    stride_sin_seq, stride_sin_model,
    # Tensor shapes
    batch_size, num_heads, seq_len, model_dim,
    # Block sizes for parallel processing
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Get program ID for parallel processing
    pid_head = tl.program_id(0)  # Head dimension
    pid_seq = tl.program_id(1)   # Sequence dimension

    # Calculate offsets
    head_idx = pid_head * BLOCK_HEAD
    seq_idx = pid_seq * BLOCK_SEQ

    # Create ranges for parallel processing
    offs_head = head_idx + tl.arange(0, BLOCK_HEAD)
    offs_seq = seq_idx + tl.arange(0, BLOCK_SEQ)
    offs_model = tl.arange(0, BLOCK_DMODEL)

    # Create masks for boundary checking
    head_mask = offs_head < num_heads
    seq_mask = offs_seq < seq_len

    # Compute base offsets for Q and K tensors
    q_offset = (offs_head[:, None, None] * stride_q_head + 
                offs_seq[None, :, None] * stride_q_seq +
                offs_model[None, None, :] * stride_q_model)
    k_offset = (offs_head[:, None, None] * stride_k_head +
                offs_seq[None, :, None] * stride_k_seq +
                offs_model[None, None, :] * stride_k_model)

    # Load Q and K values
    q = tl.load(Q_ptr + q_offset, mask=head_mask[:, None, None] & seq_mask[None, :, None])
    k = tl.load(K_ptr + k_offset, mask=head_mask[:, None, None] & seq_mask[None, :, None])

    # Load cos and sin values
    cos_offset = offs_seq[None, :, None] * stride_cos_seq + offs_model[None, None, :] * stride_cos_model
    sin_offset = offs_seq[None, :, None] * stride_sin_seq + offs_model[None, None, :] * stride_sin_model
    
    cos = tl.load(Cos_ptr + cos_offset, mask=seq_mask[None, :, None])
    sin = tl.load(Sin_ptr + sin_offset, mask=seq_mask[None, :, None])

    # Apply rotary embeddings
    # For even indices
    q_even = q[..., ::2]
    q_odd = q[..., 1::2]
    k_even = k[..., ::2]
    k_odd = k[..., 1::2]
    
    # Compute rotated values
    q_out_even = q_even * cos - q_odd * sin
    q_out_odd = q_even * sin + q_odd * cos
    k_out_even = k_even * cos - k_odd * sin
    k_out_odd = k_even * sin + k_odd * cos

    # Store results back
    q_mask = head_mask[:, None, None] & seq_mask[None, :, None]
    k_mask = head_mask[:, None, None] & seq_mask[None, :, None]
    
    tl.store(Q_ptr + q_offset[..., ::2], q_out_even, mask=q_mask)
    tl.store(Q_ptr + q_offset[..., 1::2], q_out_odd, mask=q_mask)
    tl.store(K_ptr + k_offset[..., ::2], k_out_even, mask=k_mask)
    tl.store(K_ptr + k_offset[..., 1::2], k_out_odd, mask=k_mask)

def rotary_emb_fwd(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply rotary embeddings to Q and K tensors.
    
    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, dim]
        k: Key tensor of shape [batch, num_heads, seq_len, dim]
        cos: Cosine tensor of shape [seq_len, dim]
        sin: Sine tensor of shape [seq_len, dim]
    """
    batch, num_heads, seq_len, dim = q.shape
    
    # Define block sizes
    BLOCK_HEAD = triton.next_power_of_2(min(num_heads, 32))
    BLOCK_SEQ = triton.next_power_of_2(min(seq_len, 128))
    BLOCK_DMODEL = triton.next_power_of_2(min(dim, 64))
    
    # Compute grid
    grid = (triton.cdiv(num_heads, BLOCK_HEAD), 
            triton.cdiv(seq_len, BLOCK_SEQ))
    
    # Number of warps
    num_warps = 4 if dim >= 256 else 2
    
    # Launch kernel
    _rotary_kernel[grid](
        q, k, cos, sin,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        batch, num_heads, seq_len, dim,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps
    )
    
    return q, k
