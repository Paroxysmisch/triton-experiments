import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    # Pointers to matrices
    q_ptr, k_ptr,
    # Strides for q and k
    q_row_stride, k_row_stride,
    # Pointers to cos and sin vectors
    cos_ptr, sin_ptr,
    # Strides for cos and sin
    cos_row_stride, cos_col_stride,
    sin_row_stride, sin_col_stride,
    # Matrix dimensions
    batch_size, seq_len, num_heads, head_dim,
    # Backward pass flag
    BACKWARD_PASS: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_pids = batch_size * seq_len * num_heads
    if pid >= total_pids:
        return

    # Calculate indices
    batch_idx = pid // (seq_len * num_heads)
    seq_head_idx = pid % (seq_len * num_heads)
    seq_idx = seq_head_idx // num_heads
    head_idx = seq_head_idx % num_heads

    # Bounds checking
    if batch_idx >= batch_size or seq_idx >= seq_len or head_idx >= num_heads:
        return

    # Compute pointers to the current row in q and k
    q_row_ptr = q_ptr + (batch_idx * seq_len + seq_idx) * q_row_stride
    k_row_ptr = k_ptr + (batch_idx * seq_len + seq_idx) * k_row_stride

    # Compute pointers to the current row in cos and sin
    cos_row_ptr = cos_ptr + seq_idx * cos_row_stride
    sin_row_ptr = sin_ptr + seq_idx * sin_row_stride

    # Head offset within the row
    head_offset = head_idx * head_dim

    # Process each pair of features in the head dimension
    for p in tl.static_range(head_dim // 2):
        d = 2 * p
        q_off0 = head_offset + d
        q_off1 = head_offset + d + 1
        k_off0 = head_offset + d
        k_off1 = head_offset + d + 1

        # Load cosine and sine values
        cos_val = tl.load(cos_row_ptr + p * cos_col_stride)
        sin_val = tl.load(sin_row_ptr + p * sin_col_stride)
        if BACKWARD_PASS:
            sin_val = -sin_val

        # Load query elements
        q0 = tl.load(q_row_ptr + q_off0)
        q1 = tl.load(q_row_ptr + q_off1)
        # Apply rotation
        q_rot0 = q0 * cos_val - q1 * sin_val
        q_rot1 = q0 * sin_val + q1 * cos_val
        tl.store(q_row_ptr + q_off0, q_rot0)
        tl.store(q_row_ptr + q_off1, q_rot1)

        # Load key elements
        k0 = tl.load(k_row_ptr + k_off0)
        k1 = tl.load(k_row_ptr + k_off1)
        # Apply rotation
        k_rot0 = k0 * cos_val - k1 * sin_val
        k_rot1 = k0 * sin_val + k1 * cos_val
        tl.store(k_row_ptr + k_off0, k_rot0)
        tl.store(k_row_ptr + k_off1, k_rot1)

def rope_forward(q, k, cos, sin):
    # Ensure inputs are contiguous and get dimensions
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    batch_size, seq_len, num_heads, head_dim = q.shape
    # Reshape q and k to 2D tensors (batch_size*seq_len, num_heads*head_dim)
    q_2d = q.view(batch_size * seq_len, num_heads * head_dim)
    k_2d = k.view(batch_size * seq_len, num_heads * head_dim)
    # Ensure cos and sin are correctly shaped (seq_len, head_dim//2)
    assert cos.shape == (seq_len, head_dim // 2), "cos shape mismatch"
    assert sin.shape == (seq_len, head_dim // 2), "sin shape mismatch"
    # Grid launch
    grid = (batch_size * seq_len * num_heads,)
    # Launch kernel
    _triton_rope[grid](
        q_2d, k_2d,
        q_2d.stride(0), k_2d.stride(0),
        cos, sin,
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        batch_size, seq_len, num_heads, head_dim,
        BACKWARD_PASS=False,
        BLOCK_SIZE=128,
    )
    # Reshape q and k back to original shape
    q = q_2d.view(batch_size, seq_len, num_heads, head_dim)
    k = k_2d.view(batch_size, seq_len, num_heads, head_dim)
    return q, k, cos, sin
