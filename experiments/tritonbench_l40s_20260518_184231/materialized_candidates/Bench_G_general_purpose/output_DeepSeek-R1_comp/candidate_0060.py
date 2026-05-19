import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    # Pointers to matrices
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    # Matrix dimensions
    batch_size, seq_len, n_heads, head_dim,
    # Strides for Q/K tensors
    stride_q_batch, stride_q_seq, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_seq, stride_k_head, stride_k_dim,
    # Strides for cos/sin
    stride_cos_seq, stride_cos_dim,
    stride_sin_seq, stride_sin_dim,
    # Meta-parameters
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate 3D indices (batch, seq, head)
    num_sequences = seq_len * n_heads
    batch_idx = pid // num_sequences
    seq_head_idx = pid % num_sequences
    seq_idx = seq_head_idx // n_heads
    head_idx = seq_head_idx % n_heads

    # Position in head dimension
    half_dim = head_dim // 2
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Base pointers for current sequence position
    q_base = (q_ptr + batch_idx * stride_q_batch + 
              seq_idx * stride_q_seq + head_idx * stride_q_head)
    k_base = (k_ptr + batch_idx * stride_k_batch + 
              seq_idx * stride_k_seq + head_idx * stride_k_head)
    cos_base = cos_ptr + seq_idx * stride_cos_seq
    sin_base = sin_ptr + seq_idx * stride_sin_seq

    for i in range(0, half_dim, BLOCK_SIZE):
        # Create mask to handle non-divisible BLOCK_SIZE cases
        mask = offsets + i < half_dim
        
        # Load cos/sin values for current block
        cos_offsets = offsets + i
        cos = tl.load(cos_base + cos_offsets * stride_cos_dim, mask=mask)
        sin = tl.load(sin_base + cos_offsets * stride_sin_dim, mask=mask)

        # Load Q/K values (first and second halves)
        q_offsets = cos_offsets
        q1 = tl.load(q_base + q_offsets * stride_q_dim, mask=mask)
        q2 = tl.load(q_base + (q_offsets + half_dim) * stride_q_dim, mask=mask)
        k1 = tl.load(k_base + q_offsets * stride_k_dim, mask=mask)
        k2 = tl.load(k_base + (q_offsets + half_dim) * stride_k_dim, mask=mask)

        # Apply rotation (forward or backward)
        if tl.static(BACKWARD_PASS):
            new_q1 = q1 * cos + q2 * sin
            new_q2 = -q1 * sin + q2 * cos
            new_k1 = k1 * cos + k2 * sin
            new_k2 = -k1 * sin + k2 * cos
        else:
            new_q1 = q1 * cos - q2 * sin
            new_q2 = q1 * sin + q2 * cos
            new_k1 = k1 * cos - k2 * sin
            new_k2 = k1 * sin + k2 * cos

        # Store rotated values
        tl.store(q_base + q_offsets * stride_q_dim, new_q1, mask=mask)
        tl.store(q_base + (q_offsets + half_dim) * stride_q_dim, new_q2, mask=mask)
        tl.store(k_base + q_offsets * stride_k_dim, new_k1, mask=mask)
        tl.store(k_base + (q_offsets + half_dim) * stride_k_dim, new_k2, mask=mask)

def rope_forward(q, k, cos, sin):
    # Ensure tensors are contiguous and padded
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    # Validate dimensions
    batch_size, seq_len, n_heads, head_dim = q.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    assert cos.shape == (seq_len, head_dim // 2)
    
    # Configure kernel grid and block
    grid = (batch_size * seq_len * n_heads,)
    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Limit block size
    
    # Launch kernel
    _triton_rope[grid](
        q, k, cos, sin,
        batch_size, seq_len, n_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        BACKWARD_PASS=False,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return q, k

def rope_backward(dq, dk, cos, sin):
    # Ensure gradients are contiguous and transposed if needed
    dq = dq.contiguous().transpose(1, 2)
    dk = dk.contiguous().transpose(1, 2)
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    batch_size, seq_len, n_heads, head_dim = dq.shape
    grid = (batch_size * seq_len * n_heads,)
    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    _triton_rope[grid](
        dq, dk, cos, sin,
        batch_size, seq_len, n_heads, head_dim,
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        BACKWARD_PASS=True,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dq.transpose(1, 2), dk.transpose(1, 2)
