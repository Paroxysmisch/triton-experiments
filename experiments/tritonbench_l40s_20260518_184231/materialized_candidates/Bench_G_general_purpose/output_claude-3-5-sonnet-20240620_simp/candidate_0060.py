import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    # Pointers to matrices
    q_ptr, k_ptr, cos_ptr, sin_ptr, 
    output_q_ptr, output_k_ptr,
    # Matrix dimensions
    batch_size, seq_len, num_heads, head_dim,
    # Strides for memory access
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_ob, stride_os, stride_oh, stride_od,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute pid and index
    pid = tl.program_id(0)
    num_elements = batch_size * seq_len * num_heads
    
    # Calculate batch, seq, and head indices
    batch_idx = pid // (seq_len * num_heads)
    tmp = pid % (seq_len * num_heads)
    seq_idx = tmp // num_heads
    head_idx = tmp % num_heads
    
    # Compute base pointers for this program instance
    q_base_ptr = q_ptr + batch_idx * stride_qb + seq_idx * stride_qs + head_idx * stride_qh
    k_base_ptr = k_ptr + batch_idx * stride_kb + seq_idx * stride_ks + head_idx * stride_kh
    out_q_base_ptr = output_q_ptr + batch_idx * stride_ob + seq_idx * stride_os + head_idx * stride_oh
    out_k_base_ptr = output_k_ptr + batch_idx * stride_ob + seq_idx * stride_os + head_idx * stride_oh
    
    # Load cos/sin values for this position
    cos = tl.load(cos_ptr + seq_idx)
    sin = tl.load(sin_ptr + seq_idx)
    
    # Process elements in blocks
    for dim_idx in range(0, head_dim, BLOCK_SIZE):
        # Create block mask
        mask = dim_idx + tl.arange(0, BLOCK_SIZE) < head_dim
        
        # Load query and key vectors
        q = tl.load(q_base_ptr + dim_idx * stride_qd, mask=mask)
        k = tl.load(k_base_ptr + dim_idx * stride_kd, mask=mask)
        
        # Pairs for rotation (assuming head_dim is even)
        even_indices = dim_idx + tl.arange(0, BLOCK_SIZE, 2) < head_dim
        odd_indices = dim_idx + tl.arange(1, BLOCK_SIZE, 2) < head_dim
        
        if BACKWARD_PASS:
            # Backward pass rotation
            q_rot = tl.where(even_indices, 
                           q * cos - tl.cat_right(q, -q)[1:] * sin,
                           q * cos + tl.cat_right(q, q)[:-1] * sin)
            k_rot = tl.where(even_indices,
                           k * cos - tl.cat_right(k, -k)[1:] * sin,
                           k * cos + tl.cat_right(k, k)[:-1] * sin)
        else:
            # Forward pass rotation
            q_rot = tl.where(even_indices,
                           q * cos + tl.cat_right(q, q)[1:] * sin,
                           q * cos - tl.cat_right(q, -q)[:-1] * sin)
            k_rot = tl.where(even_indices,
                           k * cos + tl.cat_right(k, k)[1:] * sin,
                           k * cos - tl.cat_right(k, -k)[:-1] * sin)
        
        # Store results
        tl.store(out_q_base_ptr + dim_idx * stride_od, q_rot, mask=mask)
        tl.store(out_k_base_ptr + dim_idx * stride_od, k_rot, mask=mask)

# Python wrapper for the kernel
def apply_rotary_pos_emb(q, k, cos, sin, backward=False):
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        cos: Cosine values of shape [seq_len]
        sin: Sine values of shape [seq_len]
        backward: Whether this is a backward pass
    
    Returns:
        Tuple of rotated (query, key) tensors
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Ensure inputs are contiguous
    q = q.contiguous()
    k = k.contiguous()
    
    # Output tensors
    output_q = torch.empty_like(q)
    output_k = torch.empty_like(k)
    
    # Calculate grid size
    grid = (batch_size * seq_len * num_heads,)
    
    # Optimal block size for the head dimension
    BLOCK_SIZE = min(triton.next_power_of_2(head_dim), 256)
    
    # Launch kernel
    _triton_rope[grid](
        q, k, cos, sin,
        output_q, output_k,
        batch_size, seq_len, num_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        output_q.stride(0), output_q.stride(1), output_q.stride(2), output_q.stride(3),
        BACKWARD_PASS=backward,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_q, output_k

def rope_backward(dq, dk, cos, sin):
    """
    Backward pass for rotary position embeddings.
    
    Args:
        dq: Query gradient of shape [batch_size, seq_len, num_heads, head_dim]
        dk: Key gradient of shape [batch_size, seq_len, num_heads, head_dim]
        cos: Cosine values of shape [seq_len]
        sin: Sine values of shape [seq_len]
    
    Returns:
        Tuple of (query_grad, key_grad)
    """
    return apply_rotary_pos_emb(dq, dk, cos, sin, backward=True)
