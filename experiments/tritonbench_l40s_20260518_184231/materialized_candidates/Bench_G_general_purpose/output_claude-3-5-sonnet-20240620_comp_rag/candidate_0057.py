import triton
import triton.language as tl
import torch

@triton.jit
def _triton_rope(
    q_ptr, k_ptr,
    cos_ptr, sin_ptr,
    stride_q, stride_k,
    stride_cos, stride_sin,
    seqlen, head_dim: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for parallel processing
    pid = tl.program_id(0)
    
    # Calculate offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_dim = head_dim // 2
    mask = col_offsets < half_dim
    
    # Load cos and sin values for current position
    pos = pid % seqlen
    cos_offset = pos * stride_cos + col_offsets
    sin_offset = pos * stride_sin + col_offsets
    cos = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0)
    
    if BACKWARD_PASS:
        sin = -sin  # Inverse rotation for backward pass
    
    # Process query
    q_offset = pid * stride_q
    q1 = tl.load(q_ptr + q_offset + col_offsets, mask=mask, other=0.0)
    q2 = tl.load(q_ptr + q_offset + col_offsets + half_dim, mask=mask, other=0.0)
    
    # Apply rotation
    q1_out = q1 * cos - q2 * sin
    q2_out = q2 * cos + q1 * sin
    
    # Store results
    tl.store(q_ptr + q_offset + col_offsets, q1_out, mask=mask)
    tl.store(q_ptr + q_offset + col_offsets + half_dim, q2_out, mask=mask)
    
    # Process key if provided
    if k_ptr is not None:
        k_offset = pid * stride_k
        k1 = tl.load(k_ptr + k_offset + col_offsets, mask=mask, other=0.0)
        k2 = tl.load(k_ptr + k_offset + col_offsets + half_dim, mask=mask, other=0.0)
        
        k1_out = k1 * cos - k2 * sin
        k2_out = k2 * cos + k1 * sin
        
        tl.store(k_ptr + k_offset + col_offsets, k1_out, mask=mask)
        tl.store(k_ptr + k_offset + col_offsets + half_dim, k2_out, mask=mask)

class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        # Save tensors for backward pass
        ctx.save_for_backward(q, k, cos, sin)
        
        # Get dimensions
        batch, seqlen, n_heads, head_dim = q.shape
        
        # Reshape inputs for kernel
        q = q.view(batch * seqlen * n_heads, head_dim)
        k = k.view(batch * seqlen * n_heads, head_dim)
        
        # Calculate optimal block size (power of 2)
        BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
        
        # Launch kernel
        grid = (q.shape[0],)  # One thread per row
        _triton_rope[grid](
            q.data_ptr(), k.data_ptr(),
            cos.data_ptr(), sin.data_ptr(),
            q.stride(0), k.stride(0),
            cos.stride(0), sin.stride(0),
            seqlen, head_dim,
            BACKWARD_PASS=False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        
        return q.view(batch, seqlen, n_heads, head_dim), k.view(batch, seqlen, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dq, dk):
        q, k, cos, sin = ctx.saved_tensors
        batch, seqlen, n_heads, head_dim = dq.shape
        
        # Reshape gradients
        dq = dq.view(batch * seqlen * n_heads, head_dim)
        dk = dk.view(batch * seqlen * n_heads, head_dim)
        
        BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
        
        # Launch kernel for backward pass
        grid = (dq.shape[0],)
        _triton_rope[grid](
            dq.data_ptr(), dk.data_ptr(),
            cos.data_ptr(), sin.data_ptr(),
            dq.stride(0), dk.stride(0),
            cos.stride(0), sin.stride(0),
            seqlen, head_dim,
            BACKWARD_PASS=True,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        
        return (
            dq.view(batch, seqlen, n_heads, head_dim),
            dk.view(batch, seqlen, n_heads, head_dim),
            None, None
        )

def apply_rope(q, k, cos, sin):
    """Wrapper function to apply rotary position embeddings"""
    return RoPEFunction.apply(q, k, cos, sin)
