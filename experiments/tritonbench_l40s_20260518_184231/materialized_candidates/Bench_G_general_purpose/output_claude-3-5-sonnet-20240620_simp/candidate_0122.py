import torch
import triton
import triton.language as tl

@triton.jit
def _rope_embedding_kernel(
    Q_ptr, cos_ptr, sin_ptr,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_cs_s, stride_cs_d,
    B, H, S, D,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute batch, head, and sequence indices
    b_idx = pid // (H * (S // BLOCK_SIZE))
    h_idx = (pid % (H * (S // BLOCK_SIZE))) // (S // BLOCK_SIZE)
    s_idx = (pid % (S // BLOCK_SIZE)) * BLOCK_SIZE
    
    # Compute base pointers
    Q_base = Q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + s_idx * stride_q_s
    cos_base = cos_ptr + s_idx * stride_cs_s
    sin_base = sin_ptr + s_idx * stride_cs_s
    
    # Load sequence mask
    seq_mask = tl.arange(0, BLOCK_SIZE) < (S - s_idx)
    
    # Process pairs of dimensions (as RoPE operates on pairs)
    for d in range(0, D, 2):
        # Load Q values for current position
        q_idx = Q_base + d
        q_real = tl.load(q_idx, mask=seq_mask)
        q_imag = tl.load(q_idx + 1, mask=seq_mask)
        
        # Load cos and sin values
        cos_val = tl.load(cos_base + d // 2 * stride_cs_d, mask=seq_mask)
        sin_val = tl.load(sin_base + d // 2 * stride_cs_d, mask=seq_mask)
        
        # Apply rotation
        q_out_real = q_real * cos_val - q_imag * sin_val
        q_out_imag = q_real * sin_val + q_imag * cos_val
        
        # Store results
        tl.store(q_idx, q_out_real, mask=seq_mask)
        tl.store(q_idx + 1, q_out_imag, mask=seq_mask)

class FastRoPEEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, cos, sin):
        # Save tensors for backward pass
        ctx.save_for_backward(q, cos, sin)
        
        # Get dimensions
        B, H, S, D = q.shape
        
        # Launch kernel
        grid = (B * H * triton.cdiv(S, 128),)
        _rope_embedding_kernel[grid](
            q, cos, sin,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            cos.stride(0), cos.stride(1),
            B, H, S, D,
            BLOCK_SIZE=128
        )
        return q

    @staticmethod
    def backward(ctx, grad_output):
        q, cos, sin = ctx.saved_tensors
        
        # Gradient for q
        grad_q = grad_output.clone()
        
        # Launch kernel with negative sin for inverse rotation
        B, H, S, D = q.shape
        grid = (B * H * triton.cdiv(S, 128),)
        _rope_embedding_kernel[grid](
            grad_q, cos, -sin,  # Note the negative sin
            grad_q.stride(0), grad_q.stride(1), grad_q.stride(2), grad_q.stride(3),
            cos.stride(0), cos.stride(1),
            B, H, S, D,
            BLOCK_SIZE=128
        )
        
        # No gradients for cos and sin as they're precomputed
        return grad_q, None, None

def fast_rope_embedding(q, k, cos, sin):
    """
    Apply RoPE to query and key tensors
    
    Args:
        q: Query tensor of shape (batch, heads, seq_len, dim)
        k: Key tensor of shape (batch, heads, seq_len, dim)
        cos: Cosine tensor of shape (seq_len, dim//2)
        sin: Sine tensor of shape (seq_len, dim//2)
    
    Returns:
        Tuple of transformed (q, k)
    """
    q = FastRoPEEmbedding.apply(q.clone(), cos, sin)
    k = FastRoPEEmbedding.apply(k.clone(), cos, sin)
    return q, k
